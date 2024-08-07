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

//For discontinuous galerkin elements
#include <deal.II/fe/fe_dgq.h>

//For moving mesh using displacement field
#include <deal.II/fe/mapping_q_eulerian.h>

//For simplices
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>



namespace NonlinearElasticity
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
			double dt;
			double end_time;
			double save_time;
			double start_time;
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
				prm.declare_entry("Save time",
					"0.005",
					Patterns::Double(),
					"Save time");
				prm.declare_entry("Start time",
					"0.0",
					Patterns::Double(),
					"Start time");
			}
			prm.leave_subsection();
		}
		void Time::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Time");
			{
				dt = prm.get_double("Timestep");
				end_time = prm.get_double("End time");
				save_time = prm.get_double("Save time");
				start_time = prm.get_double("Start time");
			}
			prm.leave_subsection();
		}
		struct Numerical
		{
			double alpha;
			double beta;
			int rk_order;
			int n_ref;
			unsigned int order;
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
				prm.declare_entry("Time integrator order",
					"1",
					Patterns::Integer(),
					"Time integrator order");
				prm.declare_entry("n_ref",
					"4",
					Patterns::Integer(),
					"n_ref");
				prm.declare_entry("Momentum order",
					"1",
					Patterns::Integer(0),
					"Momentum order");

			}
			prm.leave_subsection();
		}
		void Numerical::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				alpha = prm.get_double("alpha");
				beta = prm.get_double("beta");
				rk_order = prm.get_integer("Time integrator order");
				n_ref = prm.get_integer("n_ref");
				order = prm.get_integer("Momentum order");
			}
			prm.leave_subsection();
		}
		struct Simulation
		{
			double BodyForce;
			double InitialVelocity;
			double TractionMagnitude;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Simulation::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Simulation parameters");
			{
				prm.declare_entry("Body force",
					"-100",
					Patterns::Double(),
					"Body Force");
				prm.declare_entry("Initial velocity",
					"0",
					Patterns::Double(),
					"Initial velocity");
				prm.declare_entry("Traction Magnitude",
					"0",
					Patterns::Double(),
					"Traction Magnitude");
			}
			prm.leave_subsection();
		}
		void Simulation::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Simulation parameters");
			{
				BodyForce = prm.get_double("Body force");
				InitialVelocity = prm.get_double("Initial velocity");
				TractionMagnitude = prm.get_double("Traction Magnitude");
			}
			prm.leave_subsection();
		}
		struct AllParameters :
			public Materials,
			public Time,
			public Numerical,
			public Simulation
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
			Simulation::declare_parameters(prm);
		}
		void AllParameters::parse_parameters(ParameterHandler& prm)
		{
			Materials::parse_parameters(prm);
			Time::parse_parameters(prm);
			Numerical::parse_parameters(prm);
			Simulation::parse_parameters(prm);
		}
	} // namespace Parameters





	//Class for storing pk1 tensor values, pressure values, and cofactor values
	template <int dim>
	struct PointHistory
	{
		Tensor<2, dim> pk1_store;
		Tensor<2, dim> Cofactor_store;
		Tensor<2, dim> face_Cofactor_store;
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
		get_real_FF(const std::vector<Tensor<1, dim>>& grad_p)
	{
		Tensor<2, dim> FF;
		Tensor<2, dim> I = unit_symmetric_tensor<dim>();
		for (unsigned int i = 0; i < dim; ++i) {
			for (unsigned int j = 0; j < dim; ++j) {
				FF[i][j] = grad_p[i][j];
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
		get_Cofactor(Tensor<2, dim>& FF, double& Jf)
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
		double dimension = 1.0 * dim;
		Tensor<2, dim> strain;
		strain = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / dimension * CofactorF / Jf) + (pressure * CofactorF);
		return strain;
	}

	template <int dim>
	Tensor<2, dim> //calculates pk1 = pk1_dev+pk1_vol
		get_real_pk1(Tensor<2, dim>& FF, const double& mu, double& Jf, double& kappa, Tensor<2, dim>& HH)
	{
		Tensor<2, 3> full_FF;

		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				full_FF[i][j] = FF[i][j];
			}
		}
		if (dim == 2)
			full_FF[2][2] = 1;
		Tensor<2, 3> full_HH;
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				full_HH[i][j] = HH[i][j];
			}
		}
		if (dim == 2)
			full_HH[2][2] = Jf;

		Tensor<2, 3>  full_pk1_stress;
		Tensor<2, dim> stress;
		full_pk1_stress = full_FF;
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

		/*cout << "FF : " << full_FF<<std::endl;
		cout << "HH : " << HH<<std::endl;
		cout << "Jf : " << Jf << std::endl;
		cout << "Full HH : " << full_HH << std::endl;
		cout << "PK1 : " << full_pk1_stress <<std::endl;
		cout << std::endl;*/
		return stress;
	}

	template <int dim>
	inline Tensor<2, dim> //Provides construction of PK1 stress tensor
		get_pk1_all(Tensor<2, dim> FF, const double mu)
	{
		double Jf = get_Jf(FF);
		Tensor<2, dim> CofactorF = get_Cofactor(FF, Jf);
		Tensor<2, dim> pk1 = get_pk1(FF, mu, Jf, CofactorF);

		return pk1;
	}





	template <int dim>
	class Inelastic
	{
	public:
		Inelastic(const std::string& input_file);
		~Inelastic();
		void run();

	private:
		void         create_coarse_grid(Triangulation<2>& triangulation);
		void         create_coarse_grid(Triangulation<3>& triangulation);
		void         setup_system();
		void		 assemble_system();
		void         assemble_rhs(Vector<double>& sol_n);
		void         solve_FE();
		void         solve_ssprk2();
		void		 solve_mod_trap();
		void         solve_ssprk3();
		Vector<double> solve_mint_F(Vector<double>& sol_n, Vector<double>& sol_n_plus_1);
		void		 calculate_error(Vector<double>& sol_n, double& present_time, double& displacement_error_output, double& derivative_error_output);
		void         output_results(Vector<double>& sol_n) const;

		void do_timestep();

		void update_displacement(const Vector<double>& sol_n, const double& coeff_n, const Vector<double>& sol_n_plus, const double& coeff_n_plus);

		void setup_quadrature_point_history();

		Parameters::AllParameters parameters;

		Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler;
		FESystem<dim> fe;
		MappingFE<dim> mapping_simplex;

		AffineConstraints<double> homogeneous_constraints;

		const QGauss<dim> quadrature_formula;
		const QGauss<dim - 1> face_quadrature_formula;

		std::vector<PointHistory<dim>> quadrature_point_history;
		SparsityPattern constrained_sparsity_pattern;
		SparsityPattern unconstrained_sparsity_pattern;
		SparseMatrix<double> constrained_mass_matrix;
		SparseMatrix<double> unconstrained_mass_matrix;


		Vector<double> system_rhs;

		Vector<double> solution;
		Vector<double> old_solution;
		Vector<double> int_solution;   //For RK2 and higher order
		Vector<double> int_solution_2; //For RK3 and higher order

		Vector<double> displacement_error;
		Vector<double> derivative_error;
		Vector<double> true_solution;
		Vector<double> derivative_solution;

		Vector<double> initial_displacement;
		Vector<double> total_displacement;


		double present_time;
		double dt;
		double end_time;
		double save_time;
		double save_counter;
		unsigned int timestep_no;

		double displacement_error_output;
		double derivative_error_output;
		double E;
		double nu;


		double kappa;
		double mu;
	};

	template <int dim>
	class FFPostprocessor : public DataPostprocessorTensor<dim>
	{
	public:
		FFPostprocessor()
			:
			DataPostprocessorTensor<dim>("FF",
				update_gradients)
		{}
		virtual
			void
			evaluate_vector_field
			(const DataPostprocessorInputs::Vector<dim>& input_data,
				std::vector<Vector<double> >& computed_quantities) const override
		{
			AssertDimension(input_data.solution_gradients.size(),
				computed_quantities.size());
			Tensor<2, dim> I = unit_symmetric_tensor<dim>();
			for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
			{
				AssertDimension(computed_quantities[p].size(),
					(Tensor<2, dim>::n_independent_components));
				for (unsigned int d = 0; d < dim; ++d)
					for (unsigned int e = 0; e < dim; ++e)
						computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))]
						= input_data.solution_gradients[p][d][e];
			}
		}
	};

	// Creates RHS forcing function that pushes tissue downward depending on its distance from the y-z plane
	// i.e. "downward" gravitational force applied everywhere except at bottom of hemisphere
	template<int dim>
	class RightHandSide : public Function<dim>
	{
	public:
		virtual void rhs_vector_value(const Point<dim>& p, Tensor<1, dim>& values, double& BodyForce, double& present_time)

		{
			//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());
			values[0] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * numbers::PI * numbers::PI * std::sin(numbers::PI * present_time);
			values[1] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * numbers::PI * numbers::PI * std::sin(numbers::PI * present_time);
		}
		virtual void
			rhs_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& BodyForce, double & present_time)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				RightHandSide<dim>::rhs_vector_value(points[p], value_list[p], BodyForce, present_time);
		}
	};

	template<int dim>
	class TractionVector : public Function<dim>
	{
	public:
		virtual void traction_vector_value(const Point<dim>& /*p*/, Tensor<1, dim>& values, double& TractionMagnitude)
		{
			Assert(dim >= 2, ExcInternalError());
			values[dim - 1] = TractionMagnitude;
		}
		virtual void traction_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& TractionMagnitude)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				TractionVector<dim>::traction_vector_value(points[p], value_list[p], TractionMagnitude);
		}
	};



	template <int dim>
	class InitialMomentum : public Function<dim>
	{
	public:
		InitialMomentum(double& InitialVelocity);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
	};

	template <int dim>
	InitialMomentum<dim>::InitialMomentum(double& InitialVelocity)
		: Function<dim>(dim)
		, velocity(InitialVelocity)
	{}

	template <int dim>
	void
		InitialMomentum<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim), ExcDimensionMismatch(values.size(), dim));
		values[0] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * numbers::PI;
		values[1] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * numbers::PI;

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
	class Solution : public Function<dim>
	{
	public:
		Solution(double& present_time);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
	};

	template <int dim>
	Solution<dim>::Solution(double& present_time)
		: Function<dim>(dim),
		time(present_time)
	{}

	template <int dim>
	void
		Solution<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim), ExcDimensionMismatch(values.size(), dim));
		values[0] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * std::sin(numbers::PI * time);
		values[1] = (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * std::sin(numbers::PI * time);

	}
	template <int dim>
	void Solution<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			Solution<dim>::vector_value(points[p], value_list[p]);
	}

	template <int dim>
	class Derivative : public Function<dim>
	{
	public:
		Derivative(double& present_time);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
	};

	template <int dim>
	Derivative<dim>::Derivative(double& present_time)
		: Function<dim>(dim),
		time(present_time)
	{}

	template <int dim>
	void
		Derivative<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim), ExcDimensionMismatch(values.size(), dim));
		values[0] = numbers::PI * (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * std::cos(numbers::PI * time);
		values[1] = numbers::PI * (std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1])) * std::cos(numbers::PI * time);

	}
	template <int dim>
	void Derivative<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			Derivative<dim>::vector_value(points[p], value_list[p]);
	}

	template<int dim> // Constructor for the main class
	Inelastic<dim>::Inelastic(const std::string& input_file)
		: parameters(input_file)
		, mapping_simplex(FE_Q<dim>(parameters.order))
		, fe(FE_Q<dim>(parameters.order), dim)
		, dof_handler(triangulation)
		, quadrature_formula(parameters.order + 2)
		, face_quadrature_formula(parameters.order + 2)
		, timestep_no(0)
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
		create_coarse_grid(triangulation);
		setup_system();
		E = parameters.E;
		nu = parameters.nu;
		present_time = parameters.start_time;
		dt = parameters.dt;
		end_time = parameters.end_time;
		save_time = parameters.save_time;
		mu = get_mu<dim>(E, nu);
		kappa = get_kappa<dim>(E, nu);
		cout << "Number of components : " << fe.n_components() << " and degree : " << fe.tensor_degree() << std::endl;;
		total_displacement = 0;
		displacement_error_output = 0;
		derivative_error_output = 0;
		output_results(old_solution);
		cout << "Saving results at time : " << present_time << std::endl;
		assemble_system();

		save_counter = 1;
		while (present_time < end_time-1e-12)
			do_timestep();
	}

	template <int dim>
	void Inelastic<dim>::create_coarse_grid(Triangulation<2>& triangulation)
	{

		std::vector<Point<2>> vertices = {
			{-1.0,-1.0} , {-1.0,1.0}, {1.0, 1.0}, {1.0, -1.0} };

		const std::vector < std::array<int, GeometryInfo<2>::vertices_per_cell>>
			cell_vertices = { {{0,3,1,2}} };
		const unsigned int n_cells = cell_vertices.size();

		std::vector<CellData<2>> cells(n_cells, CellData<2>());
		for (unsigned int i = 0; i < n_cells; ++i) {
			for (unsigned int j = 0; j < cell_vertices[i].size(); ++j) {
				cells[i].vertices[j] = cell_vertices[i][j];
			}
			cells[i].material_id = 0;
		}
		triangulation.create_triangulation(vertices, cells, SubCellData());


		for (const auto& cell : triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (abs(face_center[0]) == 1.0) {
						face->set_boundary_id(1);
					}
					if (abs(face_center[1]) == 1.0) {
						face->set_boundary_id(1);
					}
				}
		cout << triangulation.n_global_levels() << std::endl;
		//GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);
		triangulation.refine_global(parameters.n_ref);

		setup_quadrature_point_history();
	}

	template <int dim>
	void Inelastic<dim>::create_coarse_grid(Triangulation<3>& triangulation)
	{
		//std::vector<Point<dim>> vertices;
		std::vector<Point<3>> vertices = {
			{0.0 , 0.0 , 0.0} , {0.0, 0.1, 0.0}, {0.0, 0.0 , 0.22} , {0.0, 0.1, 0.22}, {0.0, 0.0, 0.44}, {0.0, 0.1, 0.44},
			{0.12, 0.0, 0.11}, {0.12, 0.1, 0.11}, {0.12, 0.0, 0.295}, {0.12, 0.1, 0.295}, {0.12, 0.0, 0.48}, {0.12, 0.1, 0.48},
			{0.24, 0.0, 0.22}, {0.24, 0.1, 0.22}, {0.24, 0.0, 0.37}, {0.24, 0.1, 0.37}, {0.24, 0.0, 0.52}, {0.24, 0.1, 0.52},
			{0.36, 0.0, 0.33}, {0.36, 0.1, 0.33}, {0.36, 0.0, 0.445}, {0.36, 0.1, 0.445}, {0.36, 0.00, 0.56}, {0.36, 0.1, 0.56},
			{0.48, 0.0, 0.44}, {0.48, 0.1, 0.44}, {0.48, 0.0, 0.52}, {0.48, 0.1, 0.52}, {0.48, 0.0, 0.6}, {0.48, 0.1, 0.6 } };
		const std::vector < std::array<int, GeometryInfo<3>::vertices_per_cell>>
			cell_vertices = { {{0, 1, 2, 3, 6, 7, 8, 9}},
				{{2, 3, 4, 5, 8, 9, 10, 11}},
				{{6, 7, 8, 9, 12, 13, 14, 15}},
				{{8, 9, 10, 11, 14, 15, 16, 17}},
				{{12, 13, 14, 15, 18, 19, 20, 21}},
				{{14, 15, 16, 17, 20, 21, 22, 23}},
				{{18,19,20,21,24,25,26,27}},
				{{20,21,22,23,26,27,28,29}} };
		const unsigned int n_cells = cell_vertices.size();

		std::vector<CellData<3>> cells(n_cells, CellData<3>());
		for (unsigned int i = 0; i < n_cells; ++i) {
			for (unsigned int j = 0; j < cell_vertices[i].size(); ++j) {
				cells[i].vertices[j] = cell_vertices[i][j];
			}
			cells[i].material_id = 0;
		}
		triangulation.create_triangulation(vertices, cells, SubCellData());


		for (const auto& cell : triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (face_center[0] == 0) {
						face->set_boundary_id(4);
					}
					if (abs(face_center[0] - 0.48) < 0.015) {
						face->set_boundary_id(5);
					}
				}
		triangulation.refine_global(parameters.n_ref);
		setup_quadrature_point_history();
	}







	template <int dim>
	void Inelastic<dim>::setup_system()
	{

		dof_handler.distribute_dofs(fe);

		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

		std::cout << "Setting up zero boundary conditions" << std::endl;

		FEValuesExtractors::Vector Momentum(0);
		homogeneous_constraints.clear();

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
			dof_handler,
			1,
			Functions::ZeroFunction<dim>(dim),
			u_constraints,
			fe.component_mask(Momentum));
		u_constraints.close();

		homogeneous_constraints.merge(u_constraints);
		homogeneous_constraints.close();

		DynamicSparsityPattern dsp_constrained(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler,
			dsp_constrained,
			homogeneous_constraints,
			false);
		constrained_sparsity_pattern.copy_from(dsp_constrained);
		constrained_mass_matrix.reinit(constrained_sparsity_pattern);


		DynamicSparsityPattern dsp_unconstrained(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp_unconstrained);
		unconstrained_sparsity_pattern.copy_from(dsp_unconstrained);
		unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);



		solution.reinit(dof_handler.n_dofs());

		old_solution.reinit(dof_handler.n_dofs());

		int_solution.reinit(dof_handler.n_dofs());

		int_solution_2.reinit(dof_handler.n_dofs());

		true_solution.reinit(dof_handler.n_dofs());

		derivative_solution.reinit(dof_handler.n_dofs());

		displacement_error.reinit(dof_handler.n_dofs());
		displacement_error.reinit(dof_handler.n_dofs());

		system_rhs.reinit(dof_handler.n_dofs());

		cout << "Applying initial conditions" << std::endl;
		VectorTools::interpolate(mapping_simplex, dof_handler, InitialMomentum<dim>(parameters.InitialVelocity), old_solution);

		initial_displacement.reinit(dof_handler.n_dofs());
		total_displacement.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void Inelastic<dim>::assemble_system()
	{
		constrained_mass_matrix = 0;
		unconstrained_mass_matrix = 0;

		FEValues<dim> fe_values(mapping_simplex,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();


		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		const FEValuesExtractors::Vector Momentum(0);

		const std::vector<types::global_dof_index> dofs_per_component =
			DoFTools::count_dofs_per_fe_component(dof_handler);
		/*const unsigned int n_u = dofs_per_component[0] * dim;*/

		Tensor<1, dim> fe_val_Momentum_i;



		for (const auto& cell : dof_handler.active_cell_iterators())
		{

			cell_mass_matrix = 0;
			fe_values.reinit(cell);


			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{
				for (const unsigned int i : fe_values.dof_indices())
				{
					fe_val_Momentum_i = fe_values[Momentum].value(i, q_point);
					for (const unsigned int j : fe_values.dof_indices())
					{
						cell_mass_matrix(i, j) +=
							fe_val_Momentum_i * //Momentum terms
							fe_values[Momentum].value(j, q_point) *
							fe_values.JxW(q_point);
					}
				}
			}
			cell->get_dof_indices(local_dof_indices);
			homogeneous_constraints.distribute_local_to_global(
				cell_mass_matrix,
				local_dof_indices,
				constrained_mass_matrix);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					unconstrained_mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
				}
			}
		}
	}

	template <int dim>
	void Inelastic<dim>::assemble_rhs(Vector<double>& sol_n)
	{
		system_rhs = 0;


		FEValues<dim> fe_values(mapping_simplex,
			fe,
			quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values(mapping_simplex, 
			fe,
			face_quadrature_formula,
			update_values |
			update_gradients |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();


		Vector<double>     cell_rhs(dofs_per_cell);




		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		RightHandSide<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());

		TractionVector<dim> traction_vector;
		std::vector<Tensor<1, dim>> traction_values(n_face_q_points, Tensor<1, dim>());

		const FEValuesExtractors::Vector Momentum(0);

		Tensor<2, dim> real_FF;
		Tensor<2, dim> real_pk1;
		double sol_counter;
		double real_Jf;

		const std::vector<types::global_dof_index> dofs_per_component =
			DoFTools::count_dofs_per_fe_component(dof_handler);
		/*const unsigned int n_u = dofs_per_component[0] * dim;*/

		std::vector<std::vector<Tensor<1, dim>>> displacement_grads(quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

		Tensor<1, dim> fe_val_Momentum_i;

		//Stability parameters 
		double alpha = parameters.alpha;
		double beta = parameters.beta;



		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			PointHistory<dim>* local_quadrature_points_history =
				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
			real_FF = 0;
			cell_rhs = 0;
			fe_values.reinit(cell);

			right_hand_side.rhs_vector_value_list(fe_values.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time);


			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{

				sol_counter = 0;

				fe_values.get_function_gradients(total_displacement, displacement_grads);
				real_FF = get_real_FF(displacement_grads[q_point]);


				for (const unsigned int i : fe_values.dof_indices())
				{
					fe_val_Momentum_i = fe_values[Momentum].value(i, q_point);
					cell_rhs(i) += (-scalar_product(fe_values[Momentum].gradient(i, q_point), real_FF) +
						fe_val_Momentum_i * rhs_values[q_point]) * fe_values.JxW(q_point);
				}
			}




			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}



	template<int dim>
	void Inelastic<dim>::solve_FE()
	{
		present_time -= dt;
		assemble_rhs(old_solution);
		present_time += dt;
		const Vector<double> it_count = solve_mint_F(old_solution, solution);
		total_displacement += dt * old_solution;
		cout << std::endl;
	}


	template<int dim>
	void Inelastic<dim>::solve_ssprk2()
	{
		present_time -= dt;
		initial_displacement = total_displacement;
		assemble_rhs(old_solution);
		const Vector<double> it_count = solve_mint_F(old_solution, int_solution); //Forward Euler
		total_displacement += dt * old_solution; //Forward Euler

		present_time += dt;
		assemble_rhs(int_solution);
		const Vector<double> it_count2 = solve_mint_F(int_solution, solution); //Forward Euler
		solution = 0.5 * (solution + old_solution);  //Average

		//total_displacement = initial_displacement + dt/2.0 *(old_solution+solution); //Forward Euler

		 
		total_displacement += dt * int_solution; //Forward Euler using averaged momentum
		total_displacement = 0.5 * total_displacement + 0.5 * initial_displacement; //Average displacement

		cout << std::endl;
	}

	template<int dim>
	void Inelastic<dim>::solve_mod_trap()
	{
		initial_displacement = total_displacement;
		present_time -= dt;
		assemble_rhs(old_solution);
		const Vector<double> it_count = solve_mint_F(old_solution, int_solution);
		total_displacement += dt * int_solution; //"Backward" Euler

		cout << std::endl;
		present_time += dt;
		assemble_rhs(int_solution);
		const Vector<double> it_count2 = solve_mint_F(int_solution, solution);
		solution = 0.5 * old_solution + 0.5 * solution; //Average momentum
		total_displacement = initial_displacement +0.5*dt*(solution+old_solution); //Average displacement

		cout << std::endl;
	}

	//Assembles system, solves system, updates quad data.
	template<int dim>
	void Inelastic<dim>::solve_ssprk3()
	{
		initial_displacement = total_displacement;
		int_solution = 0;
		int_solution_2 = 0;
		present_time -= dt;
		assemble_rhs(old_solution);
		const Vector<double> it_count = solve_mint_F(old_solution, int_solution);
		total_displacement += dt * old_solution;

		cout << std::endl;
		present_time += dt;
		assemble_rhs(int_solution);
		const Vector<double> it_count2 = solve_mint_F(int_solution, int_solution_2);
		int_solution_2 = 0.75 * old_solution + 0.25 * int_solution_2;
		total_displacement += dt * int_solution;
		total_displacement = 0.75 * initial_displacement + 0.25 * total_displacement;

		cout << std::endl;
		present_time -= 0.5 * dt;
		assemble_rhs(int_solution_2);
		present_time += 0.5 * dt;
		const Vector<double> it_count3 = solve_mint_F(int_solution_2, solution);
		solution = 1.0 / 3.0 * old_solution + 2.0 / 3.0 * solution;
		total_displacement += dt * int_solution_2;
		total_displacement = 1.0 / 3.0 * initial_displacement + 2.0 / 3.0 * total_displacement;
		cout << std::endl;
	}

	//solves system using direct solver
	template <int dim>
	Vector<double> Inelastic<dim>::solve_mint_F(Vector<double>& sol_n, Vector<double>& sol_n_plus_1)
	{

		Vector<double> it_count(2);


		SparseMatrix<double>& un_M = unconstrained_mass_matrix;
		const auto op_un_M = linear_operator(un_M);
		Vector<double> un_rhs = system_rhs;
		Vector<double> old_sol = sol_n;

		un_rhs *= dt;
		un_M.vmult_add(un_rhs, old_sol);

		AffineConstraints<double> all_constraints;

		FEValuesExtractors::Vector Momentum(0);

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
			dof_handler,
			1,
			Functions::ZeroFunction<dim>(dim),
			u_constraints,
			fe.component_mask(Momentum));
		u_constraints.close();

		all_constraints.merge(u_constraints);
		all_constraints.close();
		auto setup_constrained_rhs = constrained_right_hand_side(
			all_constraints, op_un_M, un_rhs);

		Vector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M0 = constrained_mass_matrix;


		auto& momentum = sol_n_plus_1;



		Vector<double> u_rhs = rhs;

		/*SparseDirectUMFPACK M0_direct;
		M0_direct.initialize(M0);
		M0_direct.vmult(momentum, u_rhs);*/
		SolverControl            solver_control(1000, 1e-16 * system_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
		u_preconditioner.initialize(M0, 1.2);

		solver.solve(M0, momentum, u_rhs, u_preconditioner);
		it_count[0] = solver_control.last_step();


		all_constraints.distribute(sol_n_plus_1);
		it_count[1] = solver_control.last_step();
		return it_count;
	}








	template<int dim>
	void Inelastic<dim>::calculate_error(Vector<double>& total_displacement, double& present_time, double& displacement_error_output, double& derivative_error_output) 
	{
		displacement_error = 0;
		derivative_error = 0;
		VectorTools::interpolate(mapping_simplex, dof_handler, Solution<dim>(present_time), true_solution);
		VectorTools::interpolate(mapping_simplex, dof_handler, Derivative<dim>(present_time), derivative_solution);
		displacement_error = (true_solution - total_displacement);
		derivative_error = (derivative_solution - solution);
		displacement_error_output = std::max(displacement_error.l2_norm(), displacement_error_output);
		cout << "Max displacement error value : " << displacement_error_output << std::endl;
		derivative_error_output = std::max(derivative_error.l2_norm(),derivative_error_output);
		cout << "Max derivative error value : " << derivative_error_output << std::endl;
	}





	//Spits out solution into vectors then into .vtks
	template<int dim>
	void Inelastic<dim>::output_results(Vector<double>& sol_n) const
	{

		FFPostprocessor<dim> FF_out;

		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);




		std::vector<std::string> solution_names1(dim, "momentum");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation1(
				dim,
				DataComponentInterpretation::component_is_part_of_vector);


		data_out.add_data_vector(sol_n,
			solution_names1,
			DataOut<dim>::type_dof_data,
			interpretation1);
		data_out.add_data_vector(total_displacement, FF_out);
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
		data_out.add_data_vector(total_displacement, "displacement", DataOut<dim>::type_dof_data, interpretation1);
		data_out.add_data_vector(displacement_error, "Error_Per_Cell", DataOut<dim>::type_dof_data, interpretation1);
		data_out.add_data_vector(true_solution, "True_solution", DataOut<dim>::type_dof_data, interpretation1);


		std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
		GridTools::get_subdomain_association(triangulation, partition_int);

		MappingQEulerian<dim> patches_mapping(fe.tensor_degree(), dof_handler, total_displacement);
		hp::MappingCollection<dim> mapping_collection(mapping_simplex);
		mapping_collection.push_back(patches_mapping);
		data_out.build_patches(mapping_collection, 1);

		DataOutBase::VtkFlags vtk_flags;
		vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::default_compression;
		data_out.set_flags(vtk_flags);

		std::ofstream output("output-" + std::to_string(timestep_no) + ".vtu");
		data_out.write_vtu(output);
	}





	template <int dim>
	void Inelastic<dim>::do_timestep()
	{
		present_time += dt;
		++timestep_no;
		cout << "_____________________________________________________________" << std::endl;
		cout << "Timestep " << timestep_no << " at time " << present_time
			<< std::endl;
		if (present_time > end_time)
		{
			dt -= (present_time - end_time);
			present_time = end_time;
		}

		if (parameters.rk_order == 1)
		{
			solve_FE();
		}
		else if (parameters.rk_order == 2)
		{
			solve_ssprk2();
		}
		else if (parameters.rk_order == 3)
		{
			solve_ssprk3();
		}
		else if (parameters.rk_order == 4) {
			solve_mod_trap();
		}
		if (abs(present_time - save_counter * save_time) < 0.1 * dt) {
			cout << "Saving results at time : " << present_time << std::endl;
			calculate_error(total_displacement, present_time, displacement_error_output, derivative_error_output);
			output_results(solution);
			save_counter++;
		}
		std::swap(old_solution, solution);

		cout << std::endl << std::endl;
	}




	template<int dim>
	void Inelastic<dim>::update_displacement(const Vector<double>& sol_n, const double& coeff_n, const Vector<double>& sol_n_plus, const double& coeff_n_plus)
	{
		auto momentum = sol_n_plus;
		auto old_momentum = sol_n;

		total_displacement += dt * (coeff_n * old_momentum + coeff_n_plus * momentum);
	}




	// This chunk of code allows for communication between current code state and quad point history
	template<int dim>
	void Inelastic<dim>::setup_quadrature_point_history()
	{
		triangulation.clear_user_data();
		std::vector<PointHistory<dim>> tmp;
		quadrature_point_history.swap(tmp);
		quadrature_point_history.resize(triangulation.n_active_cells() * quadrature_formula.size());
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
int main(int /*argc*/, char** /*argv*/)
{
	try
	{
		using namespace dealii;
		using namespace NonlinearElasticity;

		NonlinearElasticity::Inelastic<2> inelastic("parameter_file.prm");
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

