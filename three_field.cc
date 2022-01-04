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
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iomanip>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

//for dealing with constraints for time dependent problems
#include <deal.II/lac/constrained_linear_operator.h>

//for block matrices
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

//For sparse direct solvers
#include <deal.II/lac/sparse_direct.h>

namespace Project_attempt
{
	using namespace dealii;

	/// <summary>
	/// SPACE FOR DEFINING GLOBAL VARIABLES. REPLACE WITH "PARAMETERS" ENVIRONMENT
	/// </summary>
	static double nu = 0.3;
	static double E = 7500;



	//Class for storing old Cauchy stress tensors
	template <int dim>
	struct PointHistory
	{
		Tensor<2, dim> pk1_store;
	};

	//Class for defining Kappa
	template <int dim>
	double get_kappa(const double E, const double nu) {
		double tmp;
		tmp = E / (3 * (1 - 2 * nu));
		cout << "kappa = " << tmp << std::endl;
		return tmp;
	}

	template <int dim>
	double get_mu(const double E, const double nu) {
		double tmp = E / (2 * (1 + nu));
		cout << "mu = " << tmp << std::endl;
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


	/*template <int dim>
	Tensor<2, dim>
		get_FF(const std::vector<Tensor<1, dim>>& grad_p)
	{
		Tensor<2, dim> FF;
		Tensor<2, dim> I = unit_symmetric_tensor<dim>();
		for (unsigned int i = 0; i < dim; ++i) {
			for (unsigned int j = 0; j < dim; ++j) {
				FF[i][j] = I[i][j] + grad_p[i][j];
			}
		}
		return FF;
	}*/


	template< int dim>
	double get_Jf(Tensor<2, dim>& FF)
	{
		double Jf;
		Jf = determinant(FF);
		return Jf;

	}
	template <int dim>
	Tensor<2, dim>
		get_cofactorF(Tensor<2, dim>& FF, double& Jf)
	{
		Tensor<2, dim> CofactorF;
		CofactorF = Jf * (invert(transpose(FF)));
		//cout << "cofactorF = " << CofactorF << std::endl;

		return CofactorF;
	}

	template <int dim>
	Tensor<2, dim> //calculates pk1 = pk1_dev+pk1_vol
		get_pk1(Tensor<2, dim>& FF, const double& mu, double& Jf, const double& kappa, Tensor<2, dim>& CofactorF)
	{
		Tensor<2, dim> strain;
		strain = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 3.0 * CofactorF / Jf) + (kappa * (Jf - 1) * CofactorF);
		return strain;
	}

	template <int dim>
	inline Tensor<2, dim> //Provides construction of PK1 stress tensor
		get_pk1_all(Tensor<2, dim> FF, const double mu, const double kappa)
	{
		cout << "FF = " << FF << std::endl;
		double Jf = get_Jf(FF);
		Tensor<2, dim> CofactorF = get_cofactorF(FF, Jf);
		Tensor<2, dim> pk1 = get_pk1(FF, mu, Jf, kappa, CofactorF);

		return pk1;
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
		const Point<3> axis = curl / tan_angle;
		return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
			-angle);
	}



	template <int dim>
	class Inelastic
	{
	public:
		Inelastic();
		~Inelastic();
		void run();

	private:
		void         create_coarse_grid();
		void         setup_system();
		void         assemble_system();
		void         solve_timestep();
		unsigned int solve();
		void         output_results() const;

		/* void time_integrator();
		void time_integrator_final(); */

		void do_timestep();

		//void time_integrator();

		void move_mesh();

		void setup_quadrature_point_history();

		//void update_quadrature_point_history();


		//allow for communication and identification in "parallel world"


		parallel::shared::Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler;
		MappingFE<dim> mapping;
		FESystem<dim> fe;

		AffineConstraints<double> homogeneous_constraints;

		const QGaussSimplex<dim> quadrature_formula;

		std::vector<PointHistory<dim>> quadrature_point_history;
		BlockSparsityPattern constrained_sparsity_pattern;
		BlockSparsityPattern unconstrained_sparsity_pattern;
		BlockSparseMatrix<double> constrained_mass_matrix;
		BlockSparseMatrix<double> unconstrained_mass_matrix;


		BlockVector<double> system_rhs;

		BlockVector<double> solution;
		BlockVector<double> old_solution;

		Vector<double> incremental_displacement;


		double present_time;
		double present_timestep;
		double end_time;
		unsigned int timestep_no;

		static const double youngs_modulus;
		static const double poissons_ratio;


		static const double kappa;
		static const double mu;
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
		virtual void vector_value(const Point<dim>& /*p*/, Tensor<1, dim>& values)

		{
			//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());
			Point<dim> point_1;
			values[2] = -10; //gravity? Need to check units n'at
		}
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list)
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
	class InitialMomentum : public Function<dim>
	{
	public:
		InitialMomentum();
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
	};

	template <int dim>
	InitialMomentum<dim>::InitialMomentum()
		: Function<dim>(dim+dim*dim+1)
		, velocity(0.1)
	{}

	template <int dim>
	void
		InitialMomentum<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim+dim*dim+1), ExcDimensionMismatch(values.size(), dim));
		values = 0;

		double rotator = velocity * std::sin(p[2] * M_PI / (4));

		//std::cout << " Rotation matrix" << rotation_matrix << std::endl; 
		//std::cout << " Original point: " << p << std::endl;
		//std::cout << " Rotated Point: " << pnew << std::endl;
		values(0) = 0;//-p[1] * rotator;
		values(1) = 0;// p[0] * rotator;
		values(4) = 1;
		values(8) = 1;
		values(12) = 1;
	}
	template <int dim>
	void InitialMomentum<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			InitialMomentum<dim>::vector_value(points[p], value_list[p]);
	}


	template<int dim> // Constructor for the main class, provides values for global variables
	Inelastic<dim>::Inelastic()
		: triangulation(MPI_COMM_WORLD)
		, dof_handler(triangulation)
		, mapping(FE_SimplexP<dim>(1))
		, fe(FE_SimplexP<dim>(1), dim, FE_SimplexP<dim>(1), 1, FE_SimplexP<dim, dim>(1), dim* dim)
		, quadrature_formula(fe.degree + 1)
		, present_time(0.0)
		, present_timestep(0.001)
		, end_time(1)
		, timestep_no(0)
	{}

	template <int dim>
	const double Inelastic<dim>::kappa =
		get_kappa<dim>(E, nu);

	template <int dim>
	const double Inelastic<dim>::mu =
		get_mu<dim>(E, nu);

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
		create_coarse_grid();
		//cout << "    Number of active cells:       "
		//<< triangulation.n_active_cells() << std::endl;

		setup_system();

		//cout << "    Number of degrees of freedom:     " << dof_handler.n_dofs() << std::endl;

		while (present_time < end_time)
			do_timestep();
	}


	//Classic grid generating bit. This is where the actual hollowed hemisphere is employed
	template <int dim>
	void Inelastic<dim>::create_coarse_grid()
	{
		;
		/*if (dim == 3) {
			const Point<dim> p1( -1, -1, -1 );
			const Point<dim> p2(1, 1, 1);
		}
		else {
			const Point<dim> p1(-1, 1);
			const Point<dim> p2( 1, 1 );
		}*/
		const Point<dim> p1(-1, -1, 0);
		const Point<dim> p2(1, 1, 2);
		double side = 0; // Must equal z coordinate of bottom face for dirichlet BCs to work
		std::vector<unsigned int> repetitions(dim);
		repetitions[0] = 5;
		repetitions[1] = 5;
		repetitions[2] = 10;
		GridGenerator::subdivided_hyper_rectangle_with_simplices(triangulation,
			repetitions,
			p1,
			p2,
			false);
		for (const auto& cell : triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (face_center[2] == -side) {
						face->set_boundary_id(4);
						//else if (face_center[2] == side) // Serves to disable top incremental fixed displacements. => body forces dominate
							//face->set_boundary_id(5);
					}
				}
		setup_quadrature_point_history();

	}





	template <int dim>
	void Inelastic<dim>::setup_system()
	{

		dof_handler.distribute_dofs(fe);
		DoFRenumbering::component_wise(dof_handler);

		const std::vector<types::global_dof_index> dofs_per_component =
			DoFTools::count_dofs_per_fe_component(dof_handler);
		const unsigned int n_u = dofs_per_component[0] * dim,
			n_p = dofs_per_component[dim],
			n_f = dofs_per_component[dim + 1] * dim*dim;

		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< " (" << n_u << '+' << n_p << '+' << n_f << ')' << std::endl;







		
		std::cout << "Setting up zero boundary conditions" << std::endl;
		homogeneous_constraints.clear();
		VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim+1+dim*dim),
			homogeneous_constraints); //Establishes zero BCs, 
		DoFTools::make_hanging_node_constraints(dof_handler,
			homogeneous_constraints);
		homogeneous_constraints.close();

		std::cout << "Setting up block sparsity patterns" << std::endl;
		//DynamicSparsityPattern dsp_constrained(dof_handler.n_dofs());
		BlockDynamicSparsityPattern dsp_constrained(3, 3);
		dsp_constrained.block(0, 0).reinit(n_u, n_u);
		dsp_constrained.block(1, 0).reinit(n_p, n_u);
		dsp_constrained.block(0, 1).reinit(n_u, n_p);
		dsp_constrained.block(2, 0).reinit(n_f, n_u);
		dsp_constrained.block(0, 2).reinit(n_u, n_f);
		dsp_constrained.block(1, 1).reinit(n_p, n_p);
		dsp_constrained.block(2, 2).reinit(n_f, n_f);
		dsp_constrained.block(1, 2).reinit(n_p, n_f);
		dsp_constrained.block(2, 1).reinit(n_f, n_p);
		dsp_constrained.collect_sizes();
		DoFTools::make_sparsity_pattern(dof_handler,
			dsp_constrained,
			homogeneous_constraints,
			false);
		constrained_sparsity_pattern.copy_from(dsp_constrained);
		constrained_mass_matrix.reinit(constrained_sparsity_pattern);

		//DynamicSparsityPattern dsp_unconstrained(dof_handler.n_dofs());
		//DoFTools::make_sparsity_pattern(dof_handler,
		//	dsp_unconstrained);
		//unconstrained_sparsity_pattern.copy_from(dsp_unconstrained);
		//unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);

		BlockDynamicSparsityPattern dsp_unconstrained(3, 3);
		dsp_unconstrained.block(0, 0).reinit(n_u, n_u);
		dsp_unconstrained.block(1, 0).reinit(n_p, n_u);
		dsp_unconstrained.block(0, 1).reinit(n_u, n_p);
		dsp_unconstrained.block(2, 0).reinit(n_f, n_u);
		dsp_unconstrained.block(0, 2).reinit(n_u, n_f);
		dsp_unconstrained.block(1, 1).reinit(n_p, n_p);
		dsp_unconstrained.block(2, 2).reinit(n_f, n_f);
		dsp_unconstrained.block(1, 2).reinit(n_p, n_f);
		dsp_unconstrained.block(2, 1).reinit(n_f, n_p);
		dsp_unconstrained.collect_sizes();
		DoFTools::make_sparsity_pattern(dof_handler, dsp_unconstrained);
		unconstrained_sparsity_pattern.copy_from(dsp_unconstrained);
		unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);


		std::cout << "Setting up block vectors patterns" << std::endl;

		solution.reinit(3);
		solution.block(0).reinit(n_u);
		solution.block(1).reinit(n_p);
		solution.block(2).reinit(n_f);
		solution.collect_sizes();

		old_solution.reinit(3);
		old_solution.block(0).reinit(n_u);
		old_solution.block(1).reinit(n_p);
		old_solution.block(2).reinit(n_f);
		old_solution.collect_sizes();

		system_rhs.reinit(3);
		system_rhs.block(0).reinit(n_u);
		system_rhs.block(1).reinit(n_p);
		system_rhs.block(2).reinit(n_f);
		system_rhs.collect_sizes();

		cout << " Applying initial conditions" << std::endl;
		VectorTools::interpolate(mapping, dof_handler, InitialMomentum<dim>(), solution);


		//No longer need to pass matrices and vectors through MPI communication object

		//momentum_system_rhs.reinit(dof_handler.n_dofs());
		incremental_displacement.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void Inelastic<dim>::assemble_system()
	{
		system_rhs = 0;
		//constrained_mass_matrix = 0;

		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		std::vector<Vector<double>> sol_vec(quadrature_formula.size(), Vector<double>(dim+1+dim*dim));

		const unsigned int dpc = fe.dofs_per_cell;


		std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
			quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();


		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>     cell_rhs(dofs_per_cell);


		//Defines vectors to contain values for physical parameters


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		RightHandSide<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		const FEValuesExtractors::Vector Momentum(0);
		const FEValuesExtractors::Scalar Pressure(dim);
		const FEValuesExtractors::Tensor<2> Def_Gradient(dim + 1);


		Tensor<2, dim> FF;
		Vector<double> temp_momentum(dim);
		Tensor<2, dim> Cofactor;
		double Jf;
		Tensor<2, dim> pk1;
		double vectorcounter;
		double sol_counter;

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			PointHistory<dim>* local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
			/*Assert(local_quadrature_points_history >=
				&quadrature_point_history.front(),
				ExcInternalError());
			Assert(local_quadrature_points_history <=
				&quadrature_point_history.back(),
				ExcInternalError());*/

			FF = 0;
			Cofactor = 0;
			Jf = 0;
			pk1 = 0;
			temp_momentum = 0;
			vectorcounter = 0;

			cell_mass_matrix = 0;
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values.get_function_values(solution, sol_vec);


			//creates stiffness matrix for solving linearized, isotropic elasticity equation in weak form
			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{

				sol_counter = 0;
				for (unsigned int i = 0; i < dim; i++) { //Extracts momentum values, puts them in vector form

					temp_momentum[i] = sol_vec[q_point](sol_counter);

					++sol_counter;
				}
				sol_counter += 1; //Add one to skip over pressure

				for (unsigned int i = 0; i < dim; i++) {
					for (unsigned int j = 0; j < dim; j++) { // Extracts deformation gradient values, puts them in tensor form
						FF[i][j] = sol_vec[q_point](sol_counter);
						++sol_counter;
					}
				}
				//cout << "deformation gradient values : " << FF << std::endl;
				Jf = get_Jf(FF);
				Cofactor = get_cofactorF(FF, Jf);
				//auto Cofactor_operator = linear_operator(Cofactor);
				pk1 = get_pk1(FF, mu, Jf, kappa, Cofactor);
				for (const unsigned int i : fe_values.dof_indices())
				{
					for (const unsigned int j : fe_values.dof_indices())
					{
						// For all the diagonal mass matrices
//						if ((i < dim && j < dim)) {
						cell_mass_matrix(i, j) +=
							fe_values[Momentum].value(i, q_point) *
							fe_values[Momentum].value(j, q_point) *
							fe_values.JxW(q_point);

						//						}
						//						else if (i == dim && j == dim) {
													/*cell_mass_matrix(i, j) += 1 / kappa *
														fe_values[Pressure].value(i, q_point) *
														fe_values[Pressure].value(j, q_point) *
														fe_values.JxW(q_point) +
														scalar_product(Cofactor * fe_values[Pressure].gradient(i,q_point),
														Cofactor * fe_values[Pressure].gradient(j,q_point)) *
														fe_values.JxW(q_point);*/
														//						}
														//						else if ((i > dim && j > dim)) {  // THIS SECTION NEEDS TO BE REVISED FOR MATHEMATICAL PROPRIETY
																					/*cell_mass_matrix(i,j) += scalar_product(
																						fe_values[Def_Gradient].value(i, q_point),
																						fe_values[Def_Gradient].value(j, q_point)) *
																						fe_values.JxW(q_point);*/
																						//					}

						right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);


						// NEED TO REDO THESE LINES
						//fe_values.get_function_gradients(incremental_displacement, displacement_increment_grads);
						//Tensor<2, dim> pk1 = get_pk1_all(FF, mu, kappa);

					}
					//					if (i < dim) {
					cell_rhs(i) += -scalar_product(pk1, fe_values[Momentum].gradient(i, q_point)) * fe_values.JxW(q_point) +
						fe_values[Momentum].value(i, q_point) * rhs_values[q_point] * fe_values.JxW(q_point);
					//					}
					//					else if (i > dim) {
					cell_rhs(i) += /*temp_momentum[vectorcounter] *
						fe_values[Def_Gradient].gradient(i, q_point) *
						fe_values.JxW(q_point);*/
						0;

					if (i == 0) {
						local_quadrature_points_history[q_point].pk1_store = pk1;
					}
				}
			}

			/*	const PointHistory<dim>* local_quadrature_points_data =
					reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());*/

					//for calculating the RHS with DBC: f_j^K = (f_compj,phi_j)_K - (sigma, epsilon(delta u))_K





			cell->get_dof_indices(local_dof_indices);


			homogeneous_constraints.distribute_local_to_global(
				cell_mass_matrix,
				local_dof_indices,
				constrained_mass_matrix);



			//Remove in favor of old fashioned dl2g
			//VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(dim), boundary_values);
			/*MatrixTools::apply_boundary_values(boundary_values,
				system_matrix, incremental_displacement, momentum_system_rhs,false);*/
			for (unsigned int i = 0; i < dpc; ++i) {
				for (unsigned int j = 0; j < dpc; ++j) {
					unconstrained_mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
				}
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
			system_rhs.add(local_dof_indices, cell_rhs);
		}


	FEValuesExtractors::Scalar x_component(dim - 3);
	FEValuesExtractors::Scalar y_component(dim - 2);
	FEValuesExtractors::Scalar z_component(dim - 1);





	}


//Assembles system, solves system, updates quad data.
template<int dim>
void Inelastic<dim>::solve_timestep()
{
	cout << " Assembling system..." << std::flush;
	assemble_system();
	cout << "norm of rhs is " << system_rhs.l2_norm() << std::endl;

	cout << "Attempting to solve system..." << std::endl;
	const unsigned int n_iterations = solve();
	cout << "  Solver converged in " << n_iterations << " iterations." << std::endl;

	//cout << "  Updating quadrature point data..." << std::flush;
	//update_quadrature_point_history();
	cout << std::endl;
}





//solves system using CG
template <int dim>
unsigned int Inelastic<dim>::solve()
{
	std::swap(old_solution, solution);

	const auto &un_M0 = unconstrained_mass_matrix.block(0, 0);
	const auto op_M0 = linear_operator(un_M0);
	const auto &un_M2 = unconstrained_mass_matrix.block(2, 2);
	const auto op_M2 = linear_operator(un_M2);


	const auto &M0 = constrained_mass_matrix.block(0, 0);
	const auto &M2 = constrained_mass_matrix.block(2, 2);

	 auto &un_u_rhs = system_rhs.block(0);
	 auto &un_F_rhs = system_rhs.block(2);


	auto& momentum = solution.block(0);
	auto& def_grad = solution.block(2);

	const auto old_momentum = old_solution.block(0);
	const auto old_def_grad = old_solution.block(2);

	Vector<double> old_solution_vec;
	old_solution_vec = old_solution;

	// M * _^{n+1} = dt * RHS + M * _^n
	//Scale by time step size
	un_u_rhs *= present_timestep;
	un_F_rhs *= present_timestep;

	// Add on respective mass matrix * old_ momentum to respective unconstrained RHS vector
	un_u_rhs += op_M0 * old_momentum;
	un_F_rhs += op_M2 * old_def_grad;

	 FEValuesExtractors::Vector Momentum(0);
	const FEValuesExtractors::Scalar Pressure(dim);
	const FEValuesExtractors::Vector Def_Gradient(dim + 1); //Should be Tensor<2>, but component_mask only works with symmetrics or vectors?

	AffineConstraints<double> u_constraints;
	dealii::VectorTools::interpolate_boundary_values(dof_handler,
		4,
		Functions::ZeroFunction<dim>(dim+1+dim*dim),
		u_constraints,
		fe.component_mask(Momentum));
	u_constraints.close();

	AffineConstraints<double> F_constraints;
	dealii::VectorTools::interpolate_boundary_values(dof_handler,
		4,
		Functions::ZeroFunction<dim>(dim * dim+1+dim),
		F_constraints,
		fe.component_mask(Def_Gradient));
	F_constraints.close();

	auto setup_constrained_u_rhs = constrained_right_hand_side(
		u_constraints, op_M0, un_u_rhs);
	auto setup_constrained_F_rhs = constrained_right_hand_side(
		F_constraints, op_M2, un_F_rhs);

	const std::vector<types::global_dof_index> dofs_per_component = DoFTools::count_dofs_per_fe_component(dof_handler);
	const unsigned int n_u = dofs_per_component[0] * dim,
					   n_F = dofs_per_component[dim + 1] * dim * dim;


	Vector<double> u_rhs(n_u);
	setup_constrained_u_rhs.apply(u_rhs);
	Vector<double> F_rhs(n_F);
	setup_constrained_F_rhs.apply(F_rhs);



	SolverControl            solver_control(10000000, 1e-16 * system_rhs.l2_norm());
	SolverCG<Vector<double>> solver(solver_control);

	/*PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
	u_preconditioner.initialize(M0, 1.2);*/

	PreconditionJacobi<SparseMatrix<double>> F_preconditioner;
	F_preconditioner.initialize(M2, 1.2);

	SparseDirectUMFPACK M0_direct;
	M0_direct.initialize(M0); 
	M0_direct.vmult(momentum, u_rhs);
	/*solver.solve(M0,
		momentum,
		u_rhs
		u_preconditioner);*/
	u_constraints.distribute(momentum);
	//Vector<double> dp = momentum - old_momentum;
	//cout << "change in momentum: " << dp << std::endl;

	solver.solve(M2,
		def_grad,
		F_rhs,
		F_preconditioner);
	F_constraints.distribute(def_grad);
	return solver_control.last_step();
}




//Spits out solution into vectors then into .vtks
template<int dim>
void Inelastic<dim>::output_results() const
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	auto momentum = solution.block(0);

	// Do I really need a displacement output? I don't think so
	/*std::vector<std::string> solution_names;


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
	cout << "I get to here" << std::endl;

	data_out.add_data_vector(incremental_displacement, solution_names);
	cout << "I did not get to here" << std::endl;*/

	std::vector<std::string> solution_names(dim, "momentum");
	solution_names.emplace_back("pressure");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");
	solution_names.emplace_back("F");

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation(
			dim,
			DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);

	data_out.add_data_vector(solution,
		solution_names,
		DataOut<dim>::type_dof_data,
		data_component_interpretation);
	Vector<double> norm_of_pk1(triangulation.n_active_cells());
	{
		for (auto& cell : triangulation.active_cell_iterators()) {
			Tensor<2, dim> accumulated_stress;
			for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
				accumulated_stress += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].pk1_store;
			norm_of_pk1(cell->active_cell_index()) = (accumulated_stress / quadrature_formula.size()).norm();;
		}
	}
	data_out.add_data_vector(norm_of_pk1, "norm_of_pk1");

	std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
	GridTools::get_subdomain_association(triangulation, partition_int);


	//Deals with partitioning data


	data_out.build_patches(mapping);


	DataOutBase::VtkFlags vtk_flags;
	vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::default_compression;
	data_out.set_flags(vtk_flags);
	std::ofstream output("output-" + std::to_string(present_time) + ".vtu");
	data_out.write_vtu(output);
	//DataOutBase::write_pvd_record(pvd_output, times_and_names);
}





template <int dim>
void Inelastic<dim>::do_timestep()
{
	present_time += present_timestep;
	++timestep_no;
	cout << "Timestep " << timestep_no << " at time " << present_time
		<< std::endl;
	if (present_time > end_time)
	{
		present_timestep -= (present_time - end_time);
		present_time = end_time;
	}
	solve_timestep();
	//time_integrator();
	move_mesh();
	output_results();
	cout << std::endl;
}




// Moves mesh according to vertex_displacement based on incremental_displacement function and solution of system
template< int dim>
void Inelastic<dim>::move_mesh()
{
	auto momentum = solution.block(0);

	cout << "    Moving mesh..." << std::endl;
	std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
	for (auto& cell : dof_handler.active_cell_iterators())
		for (unsigned int v = 0; v < cell->n_vertices(); ++v)
			if (vertex_touched[cell->vertex_index(v)] == false)
			{
				vertex_touched[cell->vertex_index(v)] = true;
				Point<dim> tmp_momentum;
				Point<dim> tmp_loc = cell->vertex(v);
				Point<dim> tmp;

				for (unsigned int d = 0; d < dim; ++d) {
					tmp_momentum[d] = momentum(cell->vertex_dof_index(v, d));

					// SSPRK2 in here? 
					//f^int
					tmp[d] = tmp_loc[d] + present_timestep * tmp_momentum[d];
					incremental_displacement(cell->vertex_dof_index(v, d)) += 0.5 * (-tmp_loc[d] + tmp[d] + present_timestep * tmp_momentum[d]);
					//f^n+1
				}
				//cout << "Momentum : " <<  tmp_momentum << std::endl;
				//cout << "f^int : " << tmp << std::endl;
				//cout << "difference : " << tmp - tmp_loc << std::endl;
				cell->vertex(v) = 0.5 * (tmp_loc + tmp + present_timestep * tmp_momentum);

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
/*template<int dim>
void Inelastic<dim>::update_quadrature_point_history()
{
	FEValues<dim> fe_values(mapping,
		fe,
		quadrature_formula,
		update_values | update_gradients | update_quadrature_points);
	std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
		quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));


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
}*/
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

