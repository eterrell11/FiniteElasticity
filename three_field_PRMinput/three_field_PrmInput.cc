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
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

//for dealing with constraints for time dependent problems
#include <deal.II/lac/constrained_linear_operator.h>

//for block matrices
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

//For sparse direct solvers
#include <deal.II/lac/sparse_direct.h>

//For allowing an input file to be read
#include <deal.II/base/parameter_handler.h>


namespace Project_attempt
{
	using namespace dealii;

	namespace Parameters
	{
		struct Materials
		{
			double nu;
			double E;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Materials::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Material properties");
			{
				prm.declare_entry("Young's modulus",
					"1000",
					Patterns::Double(),
					"Young's modulus");
				prm.declare_entry("Poisson's ratio",
					"0.49",
					Patterns::Double(-1.0, 0.5),
					"Poisson's ratio");
			}
			prm.leave_subsection();
		}
		void Materials::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Material properties");
			{
				nu = prm.get_double("Poisson's ratio");
				E = prm.get_double("Young's modulus");
			}
			prm.leave_subsection();
		}
		struct Time
		{
			double present_timestep;
			double end_time;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Time::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Time");
			{
				prm.declare_entry("Timestep",
					"0.001",
					Patterns::Double(),
					"Timestep");
				prm.declare_entry("End time",
					"0.5",
					Patterns::Double(),
					"End time");
			}
			prm.leave_subsection();
		}
		void Time::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Time");
			{
				present_timestep = prm.get_double("Timestep");
				end_time = prm.get_double("End time");
			}
			prm.leave_subsection();
		}
		struct Numerical
		{
			double alpha;
			double beta;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Numerical::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				prm.declare_entry("alpha",
					"0.0",
					Patterns::Double(),
					"alpha");
				prm.declare_entry("beta",
					"0.0",
					Patterns::Double(),
					"beta");
			}
			prm.leave_subsection();
		}
		void Numerical::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				alpha = prm.get_double("alpha");
				beta = prm.get_double("beta");
			}
			prm.leave_subsection();
		}
		struct AllParameters :
			public Materials,
			public Time,
			public Numerical
		{
			AllParameters(const std::string& input_file);

			static void declare_parameters(ParameterHandler& prm);

			void parse_parameters(ParameterHandler& prm);
		};

		AllParameters::AllParameters(const std::string& input_file)
		{
			ParameterHandler prm;
			declare_parameters(prm);
			prm.parse_input(input_file);
			parse_parameters(prm);
		}
		void AllParameters::declare_parameters(ParameterHandler& prm)
		{
			Materials::declare_parameters(prm);
			Time::declare_parameters(prm);
			Numerical::declare_parameters(prm);
		}
		void AllParameters::parse_parameters(ParameterHandler& prm)
		{
			Materials::parse_parameters(prm);
			Time::parse_parameters(prm);
			Numerical::parse_parameters(prm);
		}
	} // namespace Parameters





	//Class for storing pk1 tensor values, pressure values, and cofactor values
	template <int dim>
	struct PointHistory
	{
		Tensor<2, dim> pk1_store;
		double pressure_store;
		Tensor<2, dim> Cofactor_store;
	};

	//Class for defining Kappa
	template <int dim>
	double get_kappa(double& E, double& nu) {
		double tmp;
		tmp = E / (3 * (1 - 2 * nu));
		cout << "kappa = " << tmp << std::endl;
		return tmp;
	}

	//Class for defining mu
	template <int dim>
	double get_mu(double& E, double& nu) {
		double tmp = E / (2 * (1 + nu));
		cout << "mu = " << tmp << std::endl;
		return tmp;
	}

	template <int dim>
	Tensor<2, dim>
		get_realFF(const std::vector<Tensor<1, dim>>& grad_p)
	{
		Tensor<2, dim> FF;
		Tensor<2, dim> I = unit_symmetric_tensor<dim>();
		for (unsigned int i = 0; i < dim; ++i) {
			for (unsigned int j = 0; j < dim; ++j) {
				FF[i][j] = I[i][j] + grad_p[i][j];
			}
		}
		return FF;
	}

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
		get_pk1(Tensor<2, dim>& FF, const double& mu, double& Jf, double& pressure, Tensor<2, dim>& CofactorF)
	{
		Tensor<2, dim> strain;
		strain = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 3.0 * CofactorF / Jf) + (pressure * CofactorF);
		return strain;
	}

	template <int dim>
	inline Tensor<2, dim> //Provides construction of PK1 stress tensor
		get_pk1_all(Tensor<2, dim> FF, const double mu)
	{
		cout << "FF = " << FF << std::endl;
		double Jf = get_Jf(FF);
		Tensor<2, dim> CofactorF = get_cofactorF(FF, Jf);
		Tensor<2, dim> pk1 = get_pk1(FF, mu, Jf, CofactorF);

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
		Inelastic(const std::string& input_file);
		~Inelastic();
		void run();

	private:
		void         create_coarse_grid();
		void         setup_system();
		void         assemble_system();
		void		 assemble_pressure_rhs();
		void		 assemble_momentum_rhs();
		void         solve_timestep();
		Vector<double> solve_mint_F();
		unsigned int solve_p();
		unsigned int solve_m();
		void         output_results() const;

		void do_timestep();

		void move_mesh();

		void setup_quadrature_point_history();

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

		Parameters::AllParameters parameters;

		double E;
		double nu;


		double kappa;
		double mu;
	};


	// Creates RHS forcing function that pushes tissue downward depending on its distance from the y-z plane
	// i.e. "downward" gravitational force applied everywhere except at bottom of hemisphere
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
		: Function<dim>(dim + dim * dim + 1)
		, velocity(.01)
	{}

	template <int dim>
	void
		InitialMomentum<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + dim * dim + 1), ExcDimensionMismatch(values.size(), dim));
		values = 0;

		double rotator = velocity * std::sin(p[2] * M_PI / (2));

		//std::cout << " Rotation matrix" << rotation_matrix << std::endl; 
		//std::cout << " Original point: " << p << std::endl;
		//std::cout << " Rotated Point: " << pnew << std::endl;
		values(0) = 0;// -p[1] * rotator;
		values(1) = 0;// p[0] * rotator;
		values(2) = rotator;
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

	template <int dim>
	class Def_Grad_bound : public Function<dim>
	{
	public:
		Def_Grad_bound() : Function<dim>(dim + dim * dim + 1)
		{}
		void
			vector_value(const Point<dim>& p,
				Vector<double>& values) const override
		{
			values = 0;
			values(4) = 1;
			values(8) = 1;
			values(12) = 1;
		}
	};



	template<int dim> // Constructor for the main class
	Inelastic<dim>::Inelastic(const std::string& input_file)
		: parameters(input_file)
		, triangulation(MPI_COMM_WORLD)
		, dof_handler(triangulation)
		, mapping(FE_SimplexP<dim>(1))
		, fe(FE_SimplexP<dim>(1), dim, FE_SimplexP<dim>(1), 1, FE_SimplexP<dim, dim>(1), dim* dim)
		, quadrature_formula(fe.degree + 1)
		, timestep_no(0)
		, E(parameters.E)
		, nu(parameters.nu)
	{}

	/*template <int dim>
	const double Inelastic<dim>::kappa = get_kappa<dim>(parameters.E, paramters.nu);

	template <int dim>
	const double Inelastic<dim>::mu = get_mu<dim>(parameters.E, parameters.nu);*/

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
		setup_system();
		E = parameters.E;
		nu = parameters.nu;
		present_timestep = parameters.present_timestep;
		end_time = parameters.end_time;

		mu = get_mu<dim>(E, nu);
		kappa = get_kappa<dim>(E, nu);
		output_results();
		while (present_time < end_time)
			do_timestep();
	}


	//Classic grid generating bit. This is where the actual hollowed hemisphere is employed
	template <int dim>
	void Inelastic<dim>::create_coarse_grid()
	{
		const Point<dim> p1(0, 0, 0);
		const Point<dim> p2(1, 1, 1);
		double side = 0; // Must equal z coordinate of bottom face for dirichlet BCs to work
		std::vector<unsigned int> refinements(dim);
		refinements[0] = 8;
		refinements[1] = 8;
		refinements[2] = 8;
		GridGenerator::subdivided_hyper_rectangle_with_simplices(triangulation,
			refinements,
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
			n_f = dofs_per_component[dim + 1] * dim * dim;

		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< " (" << n_u << '+' << n_p << '+' << n_f << ')' << std::endl;

		std::cout << "Setting up zero boundary conditions" << std::endl;

		FEValuesExtractors::Vector Momentum(0);
		FEValuesExtractors::Scalar Pressure(dim);
		std::vector<bool> Def_Gradient(dim * dim + 1 + dim); //Should be Tensor<2>, but component_mask only works with symmetrics or vectors?
		for (unsigned int i = dim + 1; i < dim * dim + 1 + dim; ++i) {
			Def_Gradient[i] = "true";
		}
		homogeneous_constraints.clear();

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
			u_constraints,
			fe.component_mask(Momentum));
		u_constraints.close();

		AffineConstraints<double> p_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
			p_constraints,
			fe.component_mask(Pressure));
		p_constraints.close();

		AffineConstraints<double> F_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Def_Grad_bound<dim>(),
			F_constraints,
			fe.component_mask(Def_Gradient));
		F_constraints.close();
		homogeneous_constraints.merge(u_constraints);
		homogeneous_constraints.merge(p_constraints);
		homogeneous_constraints.merge(F_constraints);
		homogeneous_constraints.close();

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

		cout << "Applying initial conditions" << std::endl;
		VectorTools::interpolate(mapping, dof_handler, InitialMomentum<dim>(), solution);

		incremental_displacement.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void Inelastic<dim>::assemble_system()
	{
		system_rhs = 0;

		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		std::vector<Vector<double>> sol_vec(quadrature_formula.size(), Vector<double>(dim + 1 + dim * dim));

		const unsigned int dpc = fe.dofs_per_cell;




		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();


		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>     cell_rhs(dofs_per_cell);




		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		RightHandSide<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		const FEValuesExtractors::Vector Momentum(0);
		const FEValuesExtractors::Scalar Pressure(dim);
		const FEValuesExtractors::Tensor<2> Def_Gradient(dim + 1);

		Tensor<2, dim> realFF;
		Tensor<2, dim> FF;
		Tensor<1, dim> temp_momentum;
		double         temp_pressure;
		Tensor<2, dim> Cofactor;
		double Jf;
		Tensor<2, dim> pk1;
		double sol_counter;

		const std::vector<types::global_dof_index> dofs_per_component =
			DoFTools::count_dofs_per_fe_component(dof_handler);
		const unsigned int n_u = dofs_per_component[0] * dim;

		std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
			quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim + 1 + dim * dim));

		Tensor<1, dim> fe_val_Momentum_i;
		double fe_val_Pressure_i;
		Tensor<1, dim> fe_grad_Pressure_i;
		Tensor<2, dim> fe_val_Def_Grad_i;

		//Stability parameters 
		double alpha = parameters.alpha;
		double beta = parameters.beta;

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			PointHistory<dim>* local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
			realFF = 0;
			FF = 0;
			Cofactor = 0;
			Jf = 0;
			pk1 = 0;
			temp_momentum = 0;
			temp_pressure = 0;

			cell_mass_matrix = 0;
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values.get_function_values(solution, sol_vec);
			right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);


			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{

				sol_counter = 0;
				for (unsigned int i = 0; i < dim; i++) { //Extracts momentum values, puts them in vector form

					temp_momentum[i] = sol_vec[q_point](sol_counter);

					++sol_counter;
				}

				temp_pressure = sol_vec[q_point](sol_counter);
				sol_counter += 1;

				for (unsigned int i = 0; i < dim; i++) {
					for (unsigned int j = 0; j < dim; j++) { // Extracts deformation gradient values, puts them in tensor form
						FF[i][j] = sol_vec[q_point](sol_counter);
						++sol_counter;
					}
				}

				realFF = get_realFF(displacement_increment_grads[q_point]);
				FF += alpha * (realFF - FF);
				temp_pressure += beta * mu * (determinant(realFF) - 1 - temp_pressure / kappa);
				Jf = get_Jf(FF);
				Cofactor = get_cofactorF(FF, Jf);
				pk1 = get_pk1(FF, mu, Jf, temp_pressure, Cofactor);

				local_quadrature_points_history[q_point].pk1_store = pk1;
				local_quadrature_points_history[q_point].pressure_store = temp_pressure;
				local_quadrature_points_history[q_point].Cofactor_store = Cofactor;

				for (const unsigned int i : fe_values.dof_indices())
				{
					fe_val_Momentum_i = fe_values[Momentum].value(i, q_point);
					fe_val_Pressure_i = fe_values[Pressure].value(i, q_point);
					fe_val_Def_Grad_i = fe_values[Def_Gradient].value(i, q_point);
					fe_grad_Pressure_i = fe_values[Pressure].gradient(i, q_point);
					for (const unsigned int j : fe_values.dof_indices())
					{
						cell_mass_matrix(i, j) +=
							fe_val_Momentum_i * //Momentum terms
							fe_values[Momentum].value(j, q_point) *
							fe_values.JxW(q_point) +
							scalar_product(fe_val_Def_Grad_i, //Deformation Gradient terms
								fe_values[Def_Gradient].value(j, q_point)) *
							fe_values.JxW(q_point) +
							1 / kappa * // Pressure terms
							fe_val_Pressure_i *
							fe_values[Pressure].value(j, q_point) *
							fe_values.JxW(q_point) +
							scalar_product(Cofactor * fe_grad_Pressure_i,
								Cofactor * fe_values[Pressure].gradient(j, q_point)) *
							fe_values.JxW(q_point);
					}
					cell_rhs(i) += (-scalar_product(pk1, fe_values[Momentum].gradient(i, q_point)) +
						fe_val_Momentum_i * rhs_values[q_point] +
						-temp_momentum * fe_values[Def_Gradient].divergence(i, q_point)) * fe_values.JxW(q_point);
				}
			}

			cell->get_dof_indices(local_dof_indices);
			homogeneous_constraints.distribute_local_to_global(
				cell_mass_matrix,
				local_dof_indices,
				constrained_mass_matrix);
			for (unsigned int i = 0; i < dpc; ++i) {
				for (unsigned int j = 0; j < dpc; ++j) {
					unconstrained_mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
				}
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}

	template <int dim>
	void Inelastic<dim>::assemble_pressure_rhs()
	{
		system_rhs = 0;

		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		std::vector<Vector<double>> sol_vec(quadrature_formula.size(), Vector<double>(dim + 1 + dim * dim));

		const unsigned int dpc = fe.dofs_per_cell;


		std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
			quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();


		Vector<double>     cell_rhs(dofs_per_cell);




		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Scalar Pressure(dim);

		Tensor<2, dim> FF;
		Tensor<1, dim> temp_momentum;
		Tensor<2, dim> Cofactor;
		double Jf;
		double sol_counter;


		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			PointHistory<dim>* local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

			FF = 0;
			Jf = 0;
			temp_momentum = 0;
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values.get_function_values(solution, sol_vec);


			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{
				sol_counter = 0;
				for (unsigned int i = 0; i < dim; i++) { //Extracts momentum values, puts them in vector form

					temp_momentum[i] = sol_vec[q_point](sol_counter);

					++sol_counter;
				}
				for (unsigned int i = 0; i < dim; i++) {
					for (unsigned int j = 0; j < dim; j++) { // Extracts deformation gradient values, puts them in tensor form
						FF[i][j] = sol_vec[q_point](sol_counter);
						++sol_counter;

					}
				}
				Jf = get_Jf(FF);
				Cofactor = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q_point].Cofactor_store;
				for (const unsigned int i : fe_values.dof_indices())
				{
					cell_rhs(i) += -scalar_product(temp_momentum *
						Cofactor,
						fe_values[Pressure].gradient(i, q_point)) *
						fe_values.JxW(q_point);
				}
			}

			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dpc; ++i) {
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}

	template <int dim>
	void Inelastic<dim>::assemble_momentum_rhs()
	{
		system_rhs = 0;

		FEValues<dim> fe_values(mapping,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		std::vector<Vector<double>> sol_vec(quadrature_formula.size(), Vector<double>(dim + 1 + dim * dim));

		const unsigned int dpc = fe.dofs_per_cell;


		std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
			quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();


		Vector<double>     cell_rhs(dofs_per_cell);


		//Defines vectors to contain values for physical parameters


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Momentum(0);

		double temp_pressure;
		Tensor<2, dim> Cofactor;
		double Jf;
		double sol_counter;
		double old_pressure;


		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			PointHistory<dim>* local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

			Cofactor = 0;
			temp_pressure = 0;
			old_pressure = 0;
			sol_counter = dim;
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values.get_function_values(solution, sol_vec);


			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{
				Cofactor = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q_point].Cofactor_store;
				old_pressure = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q_point].pressure_store;

				temp_pressure = sol_vec[q_point](sol_counter);

				for (const unsigned int i : fe_values.dof_indices())
				{
					cell_rhs(i) += -scalar_product((temp_pressure - old_pressure) * Cofactor, fe_values[Momentum].gradient(i, q_point)) *
						fe_values.JxW(q_point);
				}
			}

			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dpc; ++i) {
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}


	//Assembles system, solves system, updates quad data.
	template<int dim>
	void Inelastic<dim>::solve_timestep()
	{
		cout << " Assembling system..." << std::flush;
		assemble_system();
		cout << "norm of rhs is " << system_rhs.l2_norm() << std::endl;
		cout << "Attempting to solve system..." << std::endl;
		const Vector<double> it_count = solve_mint_F();
		cout << "  Intermediate momentum solver converged in " << it_count[0] << " iterations." << std::endl;
		cout << "  Deformation gradient solver converged in " << it_count[1] << " iterations." << std::endl;
		assemble_pressure_rhs();
		const unsigned int n_iterations2 = solve_p();
		assemble_momentum_rhs();
		cout << "  Pressure solver converged in " << n_iterations2 << " iterations." << std::endl;
		const unsigned int n_iterations3 = solve_m();
		cout << "  Updated momentum solver converged in " << n_iterations3 << " iterations." << std::endl;
	}

	//solves system using direct solver
	template <int dim>
	Vector<double> Inelastic<dim>::solve_mint_F()
	{

		Vector<double> it_count(2);
		std::swap(old_solution, solution);


		BlockSparseMatrix<double>& un_M = unconstrained_mass_matrix;
		const auto op_un_M = block_operator(un_M);
		const auto& M = constrained_mass_matrix;
		BlockVector<double> un_rhs = system_rhs;
		auto& sol = solution;
		BlockVector<double> old_sol = old_solution;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, old_sol);

		AffineConstraints<double> all_constraints;

		FEValuesExtractors::Vector Momentum(0);
		FEValuesExtractors::Scalar Pressure(dim);
		std::vector<bool> Def_Gradient(dim * dim + 1 + dim); 
		for (unsigned int i = dim + 1; i < dim * dim + 1 + dim; ++i) {
			Def_Gradient[i] = "true";
		}

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
			u_constraints,
			fe.component_mask(Momentum));
		u_constraints.close();

		AffineConstraints<double> F_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Def_Grad_bound<dim>(),
			F_constraints,
			fe.component_mask(Def_Gradient));
		F_constraints.close();
		all_constraints.merge(u_constraints);
		all_constraints.merge(F_constraints);
		all_constraints.close();
		auto setup_constrained_rhs = constrained_right_hand_side(
			all_constraints, op_un_M, un_rhs);

		BlockVector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M0 = constrained_mass_matrix.block(0, 0);
		const auto& M2 = constrained_mass_matrix.block(2, 2);


		auto& momentum = solution.block(0);
		auto& def_grad = solution.block(2);



		Vector<double> u_rhs = rhs.block(0);
		Vector<double> F_rhs = rhs.block(2);

		/*SparseDirectUMFPACK M0_direct;
		M0_direct.initialize(M0);
		M0_direct.vmult(momentum, u_rhs);*/
		SolverControl            solver_control(1000, 1e-16 * system_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
		u_preconditioner.initialize(M0, 1.2);

		solver.solve(M0, momentum, u_rhs, u_preconditioner);
		it_count[0] = solver_control.last_step();

		/*SparseDirectUMFPACK M2_direct;
		M2_direct.initialize(M2);
		M2_direct.vmult(def_grad, F_rhs);*/

		PreconditionJacobi<SparseMatrix<double>> F_preconditioner;
		F_preconditioner.initialize(M2, 1.2);

		solver.solve(M2, def_grad, F_rhs, F_preconditioner);
		all_constraints.distribute(solution);
		it_count[1] = solver_control.last_step();
		return it_count;
	}


	//solves system using direct solver
	template <int dim>
	unsigned int Inelastic<dim>::solve_p()
	{
		BlockSparseMatrix<double>& un_M = unconstrained_mass_matrix;
		const auto op_un_M = block_operator(un_M);
		const auto& M = constrained_mass_matrix;
		BlockVector<double> un_rhs = system_rhs;
		auto& sol = solution;
		BlockVector<double> old_sol = old_solution;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, old_sol);


		FEValuesExtractors::Vector Momentum(0);
		const FEValuesExtractors::Scalar Pressure(dim);
		std::vector<bool> Def_Gradient(dim * dim + 1 + dim); //Should be Tensor<2>, but component_mask only works with symmetrics or vectors?
		for (unsigned int i = dim + 1; i < dim * dim + 1 + dim; ++i) {
			Def_Gradient[i] = "true";
		}

		AffineConstraints<double> p_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
			p_constraints,
			fe.component_mask(Pressure));
		p_constraints.close();

		auto setup_constrained_rhs = constrained_right_hand_side(
			p_constraints, op_un_M, un_rhs);
		BlockVector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M1 = constrained_mass_matrix.block(1, 1);


		auto& pressure = solution.block(1);



		Vector<double> p_rhs = rhs.block(1);

		/*SparseDirectUMFPACK M1_direct;
		M1_direct.initialize(M1);
		M1_direct.vmult(pressure, p_rhs);*/
		SolverControl            solver_control(1000, 1e-16 * system_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> p_preconditioner;
		p_preconditioner.initialize(M1, 1.2);

		solver.solve(M1, pressure, p_rhs, p_preconditioner);

		p_constraints.distribute(solution);


		return solver_control.last_step();

	}


	//solves system using direct solver
	template <int dim>
	unsigned int Inelastic<dim>::solve_m()
	{
		BlockSparseMatrix<double>& un_M = unconstrained_mass_matrix;
		const auto op_un_M = block_operator(un_M);
		const auto& M = constrained_mass_matrix;
		BlockVector<double> un_rhs = system_rhs;
		auto& sol = solution;
		//BlockVector<double> old_sol = old_solution;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, solution);





		const auto& M0 = constrained_mass_matrix.block(0, 0);


		auto& momentum = solution.block(0);

		FEValuesExtractors::Vector Momentum(0);
		const FEValuesExtractors::Scalar Pressure(dim);
		std::vector<bool> Def_Gradient(dim * dim + 1 + dim); //Should be Tensor<2>, but component_mask only works with symmetrics or vectors?
		for (unsigned int i = dim + 1; i < dim * dim + 1 + dim; ++i) {
			Def_Gradient[i] = "true";
		}

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping,
			dof_handler,
			4,
			Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
			u_constraints,
			fe.component_mask(Momentum));
		u_constraints.close();

		auto setup_constrained_rhs = constrained_right_hand_side(
			u_constraints, op_un_M, un_rhs);
		BlockVector<double> rhs;
		rhs.reinit(sol);
		setup_constrained_rhs.apply(rhs);

		Vector<double> u_rhs = rhs.block(0);

		/*SparseDirectUMFPACK M0_direct;
		M0_direct.initialize(M0);
		M0_direct.vmult(momentum, u_rhs);*/
		SolverControl            solver_control(1000, 1e-16 * system_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
		u_preconditioner.initialize(M0, 1.2);

		solver.solve(M0, momentum, u_rhs, u_preconditioner);
		u_constraints.distribute(solution);

		return solver_control.last_step();
	}









	//Spits out solution into vectors then into .vtks
	template<int dim>
	void Inelastic<dim>::output_results() const
	{
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);




		std::vector<std::string> solution_names1(dim, "momentum");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation1(
				dim,
				DataComponentInterpretation::component_is_part_of_vector);


		std::vector<std::string> solution_names2(1, "pressure");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation2(
				1,
				DataComponentInterpretation::component_is_scalar);


		std::vector<std::string> solution_names3(dim * dim, "deformation_gradient");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation3(
				dim * dim,
				DataComponentInterpretation::component_is_part_of_tensor);

		solution_names1.insert(solution_names1.end(), solution_names2.begin(), solution_names2.end());
		solution_names1.insert(solution_names1.end(), solution_names3.begin(), solution_names3.end());

		interpretation1.insert(interpretation1.end(), interpretation2.begin(), interpretation2.end());
		interpretation1.insert(interpretation1.end(), interpretation3.begin(), interpretation3.end());

		data_out.add_data_vector(solution,
			solution_names1,
			DataOut<dim>::type_dof_data,
			interpretation1);


		Vector<double> norm_of_pk1(triangulation.n_active_cells());
		{
			for (auto& cell : triangulation.active_cell_iterators()) {
				Tensor<2, dim> accumulated_stress;
				for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
					accumulated_stress += reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].pk1_store;
				norm_of_pk1(cell->active_cell_index()) = (accumulated_stress / quadrature_formula.size()).norm();
			}
		}
		data_out.add_data_vector(norm_of_pk1, "norm_of_pk1");

		std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
		GridTools::get_subdomain_association(triangulation, partition_int);


		data_out.build_patches(mapping);

		DataOutBase::VtkFlags vtk_flags;
		vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::default_compression;
		data_out.set_flags(vtk_flags);

		std::ofstream output("output-" + std::to_string(timestep_no) + ".vtu");
		data_out.write_vtu(output);
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
		cout << "Mesh was successfully moved " << std::endl;
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
		quadrature_point_history.resize(triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
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


}



//Establsihes namespace, calls PETSc, calls run function, and all is bueno
int main(int argc, char** argv)
{
	try
	{
		using namespace dealii;
		using namespace Project_attempt;
		Utilities::MPI::MPI_InitFinalize mpi_intitialization(argc, argv, 1);

		Project_attempt::Inelastic<3> inelastic("parameter_file.prm");
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

