#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

// For using simplices
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>



//Enables Petsc for distributing work across my whopping 4 cores
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/distributed/shared_tria.h>


//Enables the usage of a SymmetricTensor "class" and rotation matrices
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>
#include <iomanip>

namespace Project_attempt
{
	using namespace dealii;

	//Class for storing old Cauchy stress tensors
	template <int dim>
	struct PointHistory
	{
		SymmetricTensor<2, dim> old_stress;
	};

	// Template for symmetric tensor for constitutive relation.
	// Should be able to take lambda and mu as arguments for later, but might not
	// VERY TIME CONSUMING IF YOUNG'S MODULUS IS A FUNCTION OF TIME
	template <int dim>
	inline SymmetricTensor<4, dim> get_stress_strain_tensor(
		const double lambda,
		const double mu)
	{
		SymmetricTensor<4, dim> tmp;
		for (unsigned int i = 0; i < dim; ++i)
			for (unsigned int j = 0; j < dim; ++j)
				for (unsigned int k = 0; k < dim; ++k)
					for (unsigned int l = 0; l < dim; ++l)
						tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
							((i == l) && (j == k) ? mu : 0.0) +
							((i == j) && (k == l) ? lambda : 0.0));
		return tmp;
	}


	//Template for the rank-2 strain tensor for the trial functions
	template <int dim>
	inline SymmetricTensor<2, dim> get_strain(const FEValues<dim>& fe_values,
		const unsigned int shape_func,
		const unsigned int q_point)
	{
		SymmetricTensor<2, dim> tmp;
		for (unsigned int i = 0; i < dim; ++i)
			tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

		for (unsigned int i = 0; i < dim; ++i)
			for (unsigned int j = 0; j < dim; ++j)
				tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
					fe_values.shape_grad_component(shape_func, q_point, j)[i]) / 2;
		return tmp;
	}

	//Get strain tensor for the gradient of a vector-valued field
	template <int dim>
	inline SymmetricTensor<2, dim>
		get_strain(const std::vector<Tensor<1, dim>>& grad)
	{
		Assert(grad.size() == dim, ExcInternalError());
		SymmetricTensor<2, dim> strain;
		for (unsigned int i = 0; i < dim; ++i)
			strain[i][i] = grad[i][i];
		for (unsigned int i = 0; i < dim; ++i)
			for (unsigned int j = i + 1; j < dim; ++j)
				strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
		return strain;
	}

	//Tensors that describe rotation due to displacement increments. Used in updating the stress component
	// after incremental deformation
	Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>>& grad_u)
	{
		const double curl = (grad_u[1][0] - grad_u[0][1]);
		const double angle = std::atan(curl);
		return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
	}

	//Tensors that describe rotation due to displacement increments. Used in updating the stress component
	// after incremental deformation
	Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>>& grad_u)
	{
		const Point<3> curl(grad_u[2][1] - grad_u[1][2], grad_u[0][2] - grad_u[2][0], grad_u[1][0] - grad_u[0][1]);
		const double tan_angle = std::sqrt(curl * curl);
		const double angle = std::atan(tan_angle);
		//if (std::abs(angle) < 1e-9)
		//{
		//	static const double rotation[3][3] = { {1, 0 ,0 }, { 0, 1 , 0 },{0, 0, 1} };
		//	static const Tensor<2, 3> rot(rotation);
		//	return rot;
		//}
		const Point<3> axis = curl / tan_angle;
		return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
			-angle);
	}



	template <int dim>
	class Inelastic
	{
	public:
		Inelastic();
		~Inelastic();  //Provides a destructor
		void run();

	private:
		void         create_coarse_grid();
		void         setup_system();
		void         assemble_system();
		void         solve_timestep();
		unsigned int solve();
		void         output_results() const;

		void do_initial_timestep(); //first timestep is different
		void do_timestep();

		void refine_grid();
		void move_mesh();

		void setup_quadrature_point_history();

		void update_quadrature_point_history();


		//allow for communication and identification in "parallel world"


		parallel::shared::Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler;
		MappingFE<dim> mapping;
		FESystem<dim> fe;

		AffineConstraints<double> hanging_node_constraints;

		const QGaussSimplex<dim> quadrature_formula;

		std::vector<PointHistory<dim>> quadrature_point_history;
		PETScWrappers::MPI::SparseMatrix system_matrix;
		PETScWrappers::MPI::Vector solution;
		PETScWrappers::MPI::Vector system_rhs;
		Vector<double> incremental_displacement;

		double present_time;
		double present_timestep;
		double end_time;
		unsigned int timestep_no;

		MPI_Comm mpi_communicator;
		const unsigned int n_mpi_processes;
		const unsigned int this_mpi_process;
		ConditionalOStream pcout;

		IndexSet locally_owned_dofs;
		IndexSet locally_relevant_dofs;
	};

	template<int dim> //Provides function that defines lambda as a function of tissue depth 
	class Mu : public Function<dim>
	{
	public:
		Mu() : Function<dim>(1) {}
		virtual
			double value(const Point<dim>& /*p*/, unsigned int component = 0) const override
		{
			(void)component;
			double nu = 0.45;
			double E = 1000;
			double return_value = E / (2 * (1 + nu)); //definition of lambda using E and nu
			return return_value;
		}
	};

	template<int dim> //Provides function that defines mu as a function of tissue depth
	class Lambda : public Function<dim>
	{
	public:
		Lambda() : Function<dim>(1) {}
		virtual
			double value(const Point<dim>& /*p*/, unsigned int component = 0) const override
		{
			(void)component;
			double nu = 0.45;
			double E = 1000;
			double return_value = E * nu / ((1 + nu) * (1 - 2 * nu)); //definition of mu using E and nu
			return return_value;
		}
	};

	// Creates RHS forcing function that pushes tissue downward depending on its distance from the y-z plane
	// i.e. "downward" gravitational force applied everywhere except at bottom of hemisphere
	// Unused in the current code, but could be implemented for gravity, or other currently irrelevant body forces
	template<int dim>
	class RightHandSide : public Function<dim>
	{
	public:
		virtual void vector_value(const Point<dim>& /*p*/, Vector<double>& values) const override

		{
			Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());
			Point<dim> point_1;
			values(0) = -12;
		}
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points, std::vector<Vector<double>>& value_list) const override
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				RightHandSide<dim>::vector_value(points[p], value_list[p]);
		}
	};


	// Provides an incremental boundary displacement to faces on the "en-face" margin of the hemisphere.
	// Simply put, the displacements will split the displacement linearly from p0 to 0 over n timesteps
	template <int dim>
	class IncrementalBoundaryValues : public Function<dim>
	{
	public:
		IncrementalBoundaryValues(const double present_time,
			const double present_timestep, const double end_time);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
		const double present_time;
		const double present_timestep;
		const double end_time;
	};

	template <int dim>
	IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
		const double present_time,
		const double present_timestep,
		const double end_time)
		: Function<dim>(dim)
		, velocity(.08)
		, present_time(present_time)
		, present_timestep(present_timestep)
		, end_time(end_time)
	{}

	template <int dim>
	void
		IncrementalBoundaryValues<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
		values = 0;
		double angle = .1 * numbers::PI;
		const Point<3> axis(0,0,1);
		double iterative_multiplier = present_timestep / end_time;
		Tensor<2, dim> rotation_matrix = Physics::Transformations::Rotations::rotation_matrix_3d(axis,iterative_multiplier * angle);
		const Tensor<1,dim> pnew = p- rotation_matrix * p;
		//std::cout << " Rotation matrix" << rotation_matrix << std::endl; 
		//std::cout << " Original point: " << p << std::endl;
		//std::cout << " Rotated Point: " << pnew << std::endl;
		values(0) = pnew[0];
		values(1) = pnew[1];
		values(2) = pnew[2]; 
		
	}
	template <int dim>
	void IncrementalBoundaryValues<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points,
			ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
	}


	template<int dim> // Constructor for the main class
	Inelastic<dim>::Inelastic()
		: triangulation(MPI_COMM_WORLD)
		, dof_handler(triangulation)
		, mapping(FE_SimplexP<dim>(1))
		, fe(FE_SimplexP<dim>(1), dim)
		, quadrature_formula(fe.degree + 1)
		, present_time(0.0)
		, present_timestep(0.1)
		, end_time(3)
		, timestep_no(0)
		, mpi_communicator(MPI_COMM_WORLD)
		, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
		, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
		, pcout(std::cout, (this_mpi_process == 0))
	{}


	//This is a destructor.
	template <int dim>
	Inelastic<dim>::~Inelastic()
	{
		dof_handler.clear();
	}

	// Split up the run function from the grid_generator to replace refinement cycles with timesteps
	template<int dim>
	void Inelastic<dim>::run()
	{
		do_initial_timestep();
		while (present_time < end_time)
			do_timestep();
	}


	//Classic grid generating bit. This is where the actual hollowed hemisphere is employed
	template <int dim>
	void Inelastic<dim>::create_coarse_grid()
	{
		const Point<dim> p1;
		const Point<dim> p2;
		/*if (dim == 3) {
			const Point<dim> p1( -1, -1, -1 );
			const Point<dim> p2(1, 1, 1);
		}
		else {
			const Point<dim> p1(-1, 1);
			const Point<dim> p2( 1, 1 );
		}*/
		double side = 1;
		GridGenerator::subdivided_hyper_cube_with_simplices(triangulation,
								10,
								-side,
								side,
								false);
		for (const auto& cell : triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (face_center[2] == -side)
						face->set_boundary_id(4);
					else if (face_center[2] == side)
						face->set_boundary_id(5);
				}
		setup_quadrature_point_history();
		
	}





	//Standard setup template, has slight differences that allow for MPI parallelization
	template <int dim>
	void Inelastic<dim>::setup_system()
	{

		dof_handler.distribute_dofs(fe);
		locally_owned_dofs = dof_handler.locally_owned_dofs();
		DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);


		hanging_node_constraints.clear();
		VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim),
			hanging_node_constraints); //Establishes zero BCs, => must replace
		DoFTools::make_hanging_node_constraints(dof_handler,
			hanging_node_constraints);
		hanging_node_constraints.close();

		DynamicSparsityPattern dsp(locally_relevant_dofs);
		DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			hanging_node_constraints,
			false);
		SparsityTools::distribute_sparsity_pattern(dsp,
			locally_owned_dofs,
			mpi_communicator,
			locally_relevant_dofs);

		//No longer need to pass matrices and vectors through MPI communication object

		system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
		solution.reinit(locally_owned_dofs, mpi_communicator);
		system_rhs.reinit(locally_owned_dofs, mpi_communicator);
		incremental_displacement.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void Inelastic<dim>::assemble_system()
	{
		system_rhs = 0;
		system_matrix = 0;

		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();

		FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>     cell_rhs(dofs_per_cell);

		Lambda<dim> lambda; //calls Lambda values from corresponding function
		Mu<dim> mu; // calls Mu values from corresponding function

		//Defines vectors to contain valuse for physical parameters
		// std::vector<double> E(n_q_points); 
		// This term isn't really needed here, would be nice to display elasticity on mesh
		std::vector<double> lambda_values(n_q_points);
		std::vector<double> mu_values(n_q_points);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		RightHandSide<dim> right_hand_side;
		std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));

		for (const auto& cell : dof_handler.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				cell_matrix = 0;
				cell_rhs = 0;

				fe_values.reinit(cell);

				lambda.value_list(fe_values.get_quadrature_points(), lambda_values); //Actually pulls lambda value
				mu.value_list(fe_values.get_quadrature_points(), mu_values); //Actually pulls mu value

				//creates stiffness matrix for solving linearized, isotropic elasticity equation in weak form
				for (const unsigned int i : fe_values.dof_indices())
				{
					const unsigned int component_i =
						fe.system_to_component_index(i).first;
					for (const unsigned int j : fe_values.dof_indices())
					{
						const unsigned int component_j =
							fe.system_to_component_index(j).first;
						for (const unsigned int q_point : fe_values.quadrature_point_indices())
						{

							cell_matrix(i, j) +=
								((fe_values.shape_grad(i, q_point)[component_i] *
									fe_values.shape_grad(j, q_point)[component_j] *
									lambda_values[q_point]) +
									(fe_values.shape_grad(i, q_point)[component_j] *
										fe_values.shape_grad(j, q_point)[component_i] *
										mu_values[q_point]) +
									((component_i == component_j) ?
										(fe_values.shape_grad(i, q_point) *
											fe_values.shape_grad(j, q_point) *
											mu_values[q_point]) :
										0)) *
								fe_values.JxW(q_point);
						}
					}
				}
				const PointHistory<dim>* local_quadrature_points_data =
					reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

				//for calculating the RHS with DBC: f_j^K = (f_compj,phi_j)_K - (sigma, epsilon(delta u))_K
				right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
				for (const unsigned int i : fe_values.dof_indices())
				{
					const unsigned int component_i = fe.system_to_component_index(i).first;
					for (const unsigned int q_point : fe_values.quadrature_point_indices()) {
						const SymmetricTensor<2, dim>& old_stress = local_quadrature_points_data[q_point].old_stress;
						cell_rhs(i) += (rhs_values[q_point](component_i) *
							fe_values.shape_value(i, q_point) -
							old_stress * get_strain(fe_values, i, q_point)) *
							fe_values.JxW(q_point);
					}
				}
				cell->get_dof_indices(local_dof_indices);
				hanging_node_constraints.distribute_local_to_global(cell_matrix,
					cell_rhs, local_dof_indices, system_matrix, system_rhs);
			}
		system_matrix.compress(VectorOperation::add);

		system_rhs.compress(VectorOperation::add);

		FEValuesExtractors::Scalar x_component(dim - 3);
		FEValuesExtractors::Scalar y_component(dim - 2);
		FEValuesExtractors::Scalar z_component(dim - 1);

		std::map<types::global_dof_index, double> boundary_values;


		//calls the incremental boundary condition functions to define the displacement of the mesh
		VectorTools::interpolate_boundary_values(
			mapping,
			dof_handler,
			5,
			IncrementalBoundaryValues<dim>(present_time, present_timestep, end_time),
			boundary_values);
		PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
		MatrixTools::apply_boundary_values(boundary_values,
			system_matrix, tmp, system_rhs, false);
		incremental_displacement = tmp;
	}


	//Assembles system, solves system, updates quad data.
	template<int dim>
	void Inelastic<dim>::solve_timestep()
	{
		pcout << " Assembling system..." << std::flush;
		assemble_system();
		pcout << "norm of rhs is " << system_rhs.l2_norm() << std::endl;

		const unsigned int n_iterations = solve();
		pcout << "  Solver converged in " << n_iterations << " iterations." << std::endl;

		pcout << "  Updating quadrature point data..." << std::flush;
		update_quadrature_point_history();
		pcout << std::endl;
	}

	//solves system using CG
	template <int dim>
	unsigned int Inelastic<dim>::solve()
	{
		PETScWrappers::MPI::Vector distributed_incremental_displacement(
			locally_owned_dofs, mpi_communicator);
		distributed_incremental_displacement = incremental_displacement;
		SolverControl            solver_control(10000000, 1e-16 * system_rhs.l2_norm());
		PETScWrappers::SolverCG  solver(solver_control, mpi_communicator);

		PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

		solver.solve(system_matrix,
			distributed_incremental_displacement,
			system_rhs,
			preconditioner);
		incremental_displacement = distributed_incremental_displacement;
		hanging_node_constraints.distribute(incremental_displacement);


		return solver_control.last_step();
	}


	//Spits out solution into vectors then into .vtks
	template<int dim>
	void Inelastic<dim>::output_results() const
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);

		std::vector<std::string> solution_names;
		switch (dim)
		{
		case 1:
			solution_names.emplace_back("displacement");
			break;
		case 2:
			solution_names.emplace_back("x_displacement");
			solution_names.emplace_back("y_displacement");
			break;
		case 3:
			solution_names.emplace_back("x_displacement");
			solution_names.emplace_back("y_displacement");
			solution_names.emplace_back("z_displacement");
			break;
		default:
			Assert(false, ExcInternalError());
		}
		data_out.add_data_vector(incremental_displacement, solution_names);

		Vector<double> norm_of_stress(triangulation.n_active_cells());
		{
			for (auto& cell : triangulation.active_cell_iterators())
				if (cell->is_locally_owned())
				{
					SymmetricTensor<2, dim> accumulated_stress;
					for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
						accumulated_stress += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].old_stress;

					norm_of_stress(cell->active_cell_index()) = (accumulated_stress / quadrature_formula.size()).norm();

				}
				else
					norm_of_stress(cell->active_cell_index()) = -1e+20;
		}
		data_out.add_data_vector(norm_of_stress, "norm_of_stress");

		std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
		GridTools::get_subdomain_association(triangulation, partition_int);


		//Deals with partitioning data
		const Vector<double> partitioning(partition_int.begin(), partition_int.end());

		data_out.add_data_vector(partitioning, "partitioning");

		data_out.build_patches(mapping);

		const std::string pvtu_master_filename =
			data_out.write_vtu_with_pvtu_record(
				"./", "solution", timestep_no, mpi_communicator, 4);
		if (this_mpi_process == 0)
		{
			static std::vector<std::pair<double, std::string>> times_and_names;
			times_and_names.push_back(
				std::pair<double, std::string>(present_time, pvtu_master_filename));
			std::ofstream pvd_output("solution.pvd");
			DataOutBase::write_pvd_record(pvd_output, times_and_names);
		}
	}

	//The initial timestep is different in that it involves creating the mesh in the first place,
	// along with refining the grid using Kelly error estimates, similar to our homework
	template<int dim>
	void Inelastic<dim>::do_initial_timestep()
	{
		present_time += present_timestep;
		++timestep_no;
		pcout << "Timestep " << timestep_no << "at time" << present_time << std::endl;
		for (unsigned int cycle = 0; cycle < 2; ++cycle)
		{
			pcout << "Cycle " << cycle << ':' << std::endl;
			if (cycle == 0)
				create_coarse_grid();
			else
				//refine_grid();
			{
			};
			pcout << "    Number of active cells:       "
				<< triangulation.n_active_cells() << " (by partition:";
			for (unsigned int p = 0; p < n_mpi_processes; ++p)
				pcout << (p == 0 ? ' ' : '+')
				<< (GridTools::count_cells_with_subdomain_association(
					triangulation, p));
			pcout << ")" << std::endl;
			setup_system();
			pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
				<< " (by partition:";
			for (unsigned int p = 0; p < n_mpi_processes; ++p)
				pcout << (p == 0 ? ' ' : '+')
				<< (DoFTools::count_dofs_with_subdomain_association(dof_handler,
					p));
			pcout << ")" << std::endl;
			solve_timestep();
		}
		move_mesh();
		output_results();

		pcout << std::endl;
	}



	template <int dim>
	void Inelastic<dim>::do_timestep()
	{
		present_time += present_timestep;
		++timestep_no;
		pcout << "Timestep " << timestep_no << " at time " << present_time
			<< std::endl;
		if (present_time > end_time)
		{
			present_timestep -= (present_time - end_time);
			present_time = end_time;
		}
		solve_timestep();
		move_mesh();
		output_results();
		pcout << std::endl;
	}

	//Refines elements based on Kelly Error Estimate
	template <int dim>
	void Inelastic<dim>::refine_grid()
	{
		Vector<float>error_per_cell(triangulation.n_active_cells());

		KellyErrorEstimator<dim>::estimate(dof_handler,
			QGaussSimplex<dim - 1>(fe.degree + 1), std::map<types::boundary_id, const Function<dim>*>(),
			incremental_displacement,
			error_per_cell,
			ComponentMask(),
			nullptr,
			MultithreadInfo::n_threads(),
			this_mpi_process);

		const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells();


		PETScWrappers::MPI::Vector distributed_error_per_cell(mpi_communicator,
			triangulation.n_active_cells(),
			n_local_cells);
		for (unsigned int i = 0; i < error_per_cell.size(); ++i)
			if (error_per_cell(i) != 0)
				distributed_error_per_cell(i) = error_per_cell(i);
		distributed_error_per_cell.compress(VectorOperation::insert);

		error_per_cell = distributed_error_per_cell;

		GridRefinement::refine_and_coarsen_fixed_number(triangulation,
			error_per_cell,
			0.3,
			0.03);
		triangulation.execute_coarsening_and_refinement();

		setup_quadrature_point_history();
	}


	// Moves mesh according to vertex_displacement based on incremental_displacement function and solution of system
	template< int dim>
	void Inelastic<dim>::move_mesh()
	{
		pcout << "    Moving mesh..." << std::endl;
		std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
		for (auto& cell : dof_handler.active_cell_iterators())
			for (unsigned int v = 0; v < cell->n_vertices(); ++v)
				if (vertex_touched[cell->vertex_index(v)] == false)
				{
					vertex_touched[cell->vertex_index(v)] = true;
					Point<dim> vertex_displacement;
					for (unsigned int d = 0; d < dim; ++d)
						vertex_displacement[d] =
						incremental_displacement(cell->vertex_dof_index(v, d));
					cell->vertex(v) += vertex_displacement;
				}
	}


    // This chunk of code allows for communication between current code state and quad point history
	template<int dim>
	void Inelastic<dim>::setup_quadrature_point_history()
	{
		triangulation.clear_user_data();
		{
			std::vector<PointHistory<dim>> tmp;
			quadrature_point_history.swap(tmp);
		}
		quadrature_point_history.resize(triangulation.n_locally_owned_active_cells()
			* quadrature_formula.size());
		unsigned int history_index = 0;
		for (auto& cell : triangulation.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				cell->set_user_pointer(&quadrature_point_history[history_index]);
				history_index += quadrature_formula.size();
			}

		Assert(history_index == quadrature_point_history.size(),
			ExcInternalError());
	}


	//Provides an update to the stress tensor using previous gradient data. This is mainly used in the RHS of assemble_system
	template<int dim>
	void Inelastic<dim>::update_quadrature_point_history()
	{
		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values | update_gradients | update_quadrature_points);
		std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
			quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));


		Lambda<dim> lambda;
		Mu<dim> mu;


		std::vector<double> lambda_values(quadrature_formula.size());
		std::vector<double> mu_values(quadrature_formula.size());
		for (auto& cell : dof_handler.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				PointHistory<dim>* local_quadrature_points_history =
					reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
				Assert(local_quadrature_points_history >=
					&quadrature_point_history.front(),
					ExcInternalError());
				Assert(local_quadrature_points_history <=
					&quadrature_point_history.back(),
					ExcInternalError());

				fe_values.reinit(cell);
				fe_values.get_function_gradients(incremental_displacement,
					displacement_increment_grads);

				lambda.value_list(fe_values.get_quadrature_points(), lambda_values); //Actually pulls lambda value
				mu.value_list(fe_values.get_quadrature_points(), mu_values); //Actually pulls mu value


				for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
				{
					const SymmetricTensor<4, dim> stress_strain_tensor =
						get_stress_strain_tensor<dim>(lambda_values[q], mu_values[q]);
					const SymmetricTensor<2, dim> new_stress =
						(local_quadrature_points_history[q].old_stress +
							(stress_strain_tensor * get_strain(displacement_increment_grads[q])));
					const Tensor<2, dim> rotation = get_rotation_matrix(displacement_increment_grads[q]);

					const SymmetricTensor<2, dim> rotated_new_stress = symmetrize(transpose(rotation) *
						static_cast<Tensor<2, dim>>(new_stress) *
						rotation);

					local_quadrature_points_history[q].old_stress = rotated_new_stress;

				}
			}
	}
}



//Establsihes namespace, calls PETSc, calls run function, and all is bueno
int main(int argc, char** argv)
{
	try
	{
		using namespace dealii;
		using namespace Project_attempt;
		Utilities::MPI::MPI_InitFinalize mpi_intitialization(argc, argv, 1);

		Project_attempt::Inelastic<3> inelastic;
		inelastic.run();
	}
	catch (std::exception& exc)
	{
		std::cerr << std::endl
			<< std::endl
			<< "---------------"
			<< std::endl;
		std::cerr << "Exception on processing:" << std::endl
			<< exc.what() << std::endl
			<< "Aborting!" << std::endl
			<< "------------"
			<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl
			<< std::endl
			<< "---------------"
			<< std::endl;
		std::cerr << "Unknown Exception!" << std::endl
			<< "Aborting!" << std::endl
			<< "------------"
			<< std::endl;

		return 1;
	}
	return 0;
}

