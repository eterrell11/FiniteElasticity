#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
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
#include <deal.II/fe/FE.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_bubbles.h>
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

//for ILU preconditioner
#include <deal.II/lac/sparse_ilu.h>

//For allowing an input file to be read
#include <deal.II/base/parameter_handler.h>

//For discontinuous galerkin elements
#include <deal.II/fe/fe_dgq.h>

//For simplices
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>



//For synchronous iterator support
#include <deal.II/base/synchronous_iterator.h>


namespace NonlinearElasticity
{
	using namespace dealii;

	namespace Parameters
	{
		struct Materials
		{
			double nu;
			double E;
			double rho_0;
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
				prm.declare_entry("Density",
					"1",
					Patterns::Double(),
					"Density");
			}
			prm.leave_subsection();
		}
		void Materials::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Material properties");
			{
				nu = prm.get_double("Poisson's ratio");
				E = prm.get_double("Young's modulus");
				rho_0 = prm.get_double("Density");

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
			int integrator;
			int n_ref;
			unsigned int velocity_order;
			unsigned int pressure_order;
			double tau_FFp;
			bool LumpMass;
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
				prm.declare_entry("Time integrator",
					"1",
					Patterns::Integer(),
					"Time integrator");
				prm.declare_entry("n_ref",
					"4",
					Patterns::Integer(),
					"n_ref");
				prm.declare_entry("Velocity order",
					"1",
					Patterns::Integer(0),
					"Velocity order");
				prm.declare_entry("Pressure order",
					"1",
					Patterns::Integer(0),
					"Pressure order");
				prm.declare_entry("Tau_FFp",
					"0.01",
					Patterns::Double(),
					"Tau_FFp");
				prm.declare_entry("LumpMass",
					"false",
					Patterns::Bool(),
					"LumpMass");

			}
			prm.leave_subsection();
		}
		void Numerical::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				alpha = prm.get_double("alpha");
				beta = prm.get_double("beta");
				integrator = prm.get_integer("Time integrator");
				n_ref = prm.get_integer("n_ref");
				velocity_order = prm.get_integer("Velocity order");
				pressure_order = prm.get_integer("Pressure order");
				tau_FFp = prm.get_double("Tau_FFp");
				LumpMass = prm.get_bool("LumpMass");
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
					"1.0",
					Patterns::Double(),
					"Body Force");
				prm.declare_entry("Initial velocity",
					"1.0",
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



	template <int dim, int spacedim = dim>
	Quadrature<dim> compute_nodal_quadrature(const FiniteElement<dim, spacedim>& fe)
	{
		//Assert(fe.n_blocks() == 1, ExcNotImplemented());
		//Assert(fe.n_components() == 1, ExcNotImplemented());
		//Needs to be called for each distinct finite element scenario
		ReferenceCell type = fe.reference_cell(); //Defines reference cell for given finite element space

		const FEValuesExtractors::Scalar Velocity0(0);
		const FEValuesExtractors::Scalar Velocity1(1);
		const FEValuesExtractors::Scalar Pressure(dim);

		// All quad points for the whole system
		Quadrature<dim> q_gauss = type.get_gauss_type_quadrature<dim>(fe.tensor_degree() + 1);
		const std::vector<Point<dim>> q_points = q_gauss.get_points();

		// All quad points corresponding to velocity
		Quadrature<dim> v_q_gauss = type.get_gauss_type_quadrature<dim>(fe.get_sub_fe(fe.component_mask(Velocity0)).tensor_degree()+1); 
		const std::vector<Point<dim>> v_q_points = v_q_gauss.get_points();
        // All quad points corresponding to pressure
		Quadrature<dim> p_q_gauss = type.get_gauss_type_quadrature<dim>(fe.get_sub_fe(fe.component_mask(Pressure)).tensor_degree() + 1);
		const std::vector<Point<dim>> p_q_points = p_q_gauss.get_points();
	
		Triangulation<dim, spacedim> tria;
		GridGenerator::reference_cell(tria, type); //make grid for reference cell
		//const Mapping<dim, spacedim>& mapping = type.template get_default_linear_mapping<dim, spacedim>();
		auto cell = tria.begin_active();
		const MappingFE<dim> mapping_simplex(FE_SimplexP_Bubbles<dim>(1));

		FE_SimplexP_Bubbles<dim> fe_velocity1(2);

		cout << fe.tensor_degree() + 1 << std::endl;

		cout << q_gauss.size() << std::endl;
		cout << v_q_gauss.size() << std::endl;
		cout << p_q_gauss.size() << std::endl;
		cout << std::endl;
		
		


		FEValues<dim, spacedim> fe_values(
			mapping_simplex,
			fe,
			q_gauss,
			update_values |
			update_quadrature_points |
			update_JxW_values);

		FEValues<dim, spacedim> fe_v_values(
			mapping_simplex,
			fe.get_sub_fe(fe.component_mask(Pressure)),
			p_q_gauss,
			update_values |
			update_quadrature_points |
			update_JxW_values);
		
		FEValues<dim, spacedim> fe_p_values(
			mapping_simplex,
			fe.get_sub_fe(fe.component_mask(Velocity1)),
			p_q_gauss,
			update_values |
			update_quadrature_points |
			update_JxW_values);


		fe_values.reinit(cell);
		fe_p_values.reinit(cell);
		fe_v_values.reinit(cell);
		std::vector<Point<dim>> nodal_quad_points = fe.get_unit_support_points(); //Find nodal locations to use as quadrature points
		std::vector<double> nodal_quad_weights(nodal_quad_points.size()); //Preallocate vector of nodal quadrature weights

		Assert(nodal_quad_points.size() > 0, ExcNotImplemented());

		//Switched q and i loops
		for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
		{
			double integral = 0;
			//bool q_flag = 0;
			//for (unsigned int j=0; j < p_q_gauss.size(); ++j) {
			//	//cout << q_points[q] << std::endl;
			//	//cout << p_q_points[j] << std::endl;
			//	//cout << std::endl;
			//	if (q_points[q] == p_q_points[j]) {
			//		q_flag = true;
			//	}
			//}
			//cout << std::find(v_q_points[0], v_q_points[v_q_gauss.size()], q_gauss.quadrature_points(q)) << std::endl;;
			for (unsigned int q = 0; q < v_q_gauss.size(); ++q)
			{
				integral += (fe_values[Velocity0].value(i, q) + fe_values[Velocity1].value(i, q)) * fe_v_values.JxW(q);

			}
			for (unsigned int q = 0; q < p_q_gauss.size(); ++q)
			{
				integral += fe_values[Pressure].value(i, q) * fe_p_values.JxW(q);

			}
			nodal_quad_weights[i] = integral;//Quadrature weights are determined by the integral computed via exact Gaussian quadrature on the reference element
			//cout << integral << std::endl;
		}

		
		return { nodal_quad_points, nodal_quad_weights };
	}



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
		get_real_FF(const Tensor<2, dim>& grad_p)
	{
		Tensor<2, dim> FF;
		Tensor<2, dim> I = unit_symmetric_tensor<dim>();
		FF = I + grad_p;
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
		get_HH(Tensor<2, dim>& FF, double& Jf)
	{
		Tensor<2, dim> HHF;
		HHF = Jf * (invert(transpose(FF)));
		//cout << "cofactorF = " << HHF << std::endl;

		return HHF;
	}

	template <int dim>
	Tensor<2, dim> //calculates pk1 = pk1_dev+pk1_vol
		get_pk1(Tensor<2, dim>& FF, const double& mu, double& Jf, double& pressure, Tensor<2, dim>& HH)
	{
		Tensor<2, 3> full_FF;
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				full_FF[i][j] = FF[i][j];
			}
		}
		if (dim == 2) {
			full_FF[2][2] = 1;
		}
		Tensor<2, 3> full_HH;
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				full_HH[i][j] = HH[i][j];
			}
		}
		if (dim == 2) {
			full_HH[2][2] = Jf;
		}
		Tensor<2, dim> stress;
		Tensor<2, 3> full_pk1_stress;
		full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3 * full_HH / Jf) + (pressure * full_HH);

		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

		//stress = mu * FF + (pressure * HH);
		//stress = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 2.0 * HH / Jf) + (pressure *HH);


		/*cout << "FF : " << full_FF<<std::endl;
		cout << "HH : " << HH<<std::endl;
		cout << "Jf : " << Jf << std::endl;*/
		/*cout << "HH : " << HH << std::endl;
		cout << "PK1 : " << stress <<std::endl;
		cout << std::endl;*/

		return stress;
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
		//stress = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 2.0 * HH / Jf) + kappa*((Jf-1) * HH);
		full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3.0 * full_HH / Jf) + kappa * ((Jf - 1) * full_HH);
		//full_pk1_stress =  mu*full_FF + kappa * ((Jf - 1) * full_HH);
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

		//stress = mu * FF + kappa * ((Jf - 1) * HH);
		return stress;
	}




	template <int dim>
	class Incompressible
	{
	public:
		Incompressible(const std::string& input_file);
		~Incompressible();
		void run();

	private:
		void         create_coarse_grid(Triangulation<2>& triangulation);
		void         create_coarse_grid(Triangulation<3>& triangulation);
		void         setup_system();
		void         assemble_system_Kuu();
		void         assemble_system_not_Kuu();
		void		 assemble_Rp();
		void         solve_semi_implicit();
		void		 solve_explicit();
		void		 solve_SI_system();
		void		 solve_FE_system();
		void		 update_motion();
		void         output_results() const;
		void		 calculate_error();
		void		 do_timestep();




		Parameters::AllParameters parameters;

		Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler;
		FESystem<dim> fe;

		MappingFE<dim> mapping_simplex;


		AffineConstraints<double> constraints;
		AffineConstraints<double> displacement_constraints;
		AffineConstraints<double> pressure_constraints;

		const QGaussSimplex<dim> quadrature_formula;
		const QGaussSimplex<dim - 1> face_quadrature_formula;



		BlockSparsityPattern sparsity_pattern;
		BlockSparsityPattern un_sparsity_pattern;
		BlockSparseMatrix<double> K;
		BlockSparseMatrix<double> un_K;


		BlockVector<double> R;
		BlockVector<double> un_R;


		BlockVector<double> solution;
		BlockVector<double> solution_increment;



		//Vector<double> residual;

		Vector<double> displacement;
		Vector<double> old_displacement;
		Vector<double> velocity;
		Vector<double> old_velocity;
		Vector<double> acceleration;
		Vector<double> old_acceleration;
		Vector<double> pressure;
		Vector<double> old_pressure;


		BlockVector<double> true_solution;
		BlockVector<double> error;

		double present_time;
		double dt;
		double rho_0;
		double end_time;
		double save_time;
		double save_counter;
		unsigned int timestep_no;
		double pressure_mean;


		Vector<double> u_cell_wise_error;
		Vector<double> p_cell_wise_error;

		double displacement_error_output;
		double velocity_error_output;
		double pressure_error_output;

		double E;
		double nu;


		double kappa;
		double mu;

		double tau;
	};






	template<int dim>
	class FExt : public Function<dim>
	{
	public:
		virtual void rhs_vector_value(const Point<dim>& /*p*/, Tensor<1, dim>& values, double& a, double& present_time, double& /*mu*/, double& /*kappa*/)

		{
			//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());

			values[0] = a * present_time;
			values[1] = 0;
		}
		virtual void
			rhs_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& BodyForce, double& present_time, double& mu, double& kappa)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				FExt<dim>::rhs_vector_value(points[p], value_list[p], BodyForce, present_time, mu, kappa);
		}
	};

	template<int dim>
	class PressureRightHandSide : public Function<dim>
	{
	public:
		virtual void rhs_value(const Point<dim>& p, double& value, double& a, double& present_time, double& mu, double& kappa)

		{
			//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());
			value = 0;
		}
		virtual void
			rhs_value_list(const std::vector<Point<dim>>& points, std::vector<double>& value_list, double& BodyForce, double& present_time, double& mu, double& kappa)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				PressureRightHandSide<dim>::rhs_value(points[p], value_list[p], BodyForce, present_time, mu, kappa);
		}
	};

	template<int dim>
	class TractionVector : public Function<dim>
	{
	public:
		virtual void traction_vector_value(const Point<dim>& /*p*/, Tensor<1, dim>& values, double& TractionMagnitude, double& time)
		{
			Assert(dim >= 2, ExcInternalError());
			values[0] = TractionMagnitude * time;
		}
		virtual void traction_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& TractionMagnitude, double& time)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				TractionVector<dim>::traction_vector_value(points[p], value_list[p], TractionMagnitude, time);
		}
	};



	template <int dim>
	class InitialSolution : public Function<dim>
	{
	public:
		InitialSolution(double& InitialVelocity, double& mu);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
		const double mu;
	};

	template <int dim>
	InitialSolution<dim>::InitialSolution(double& InitialVelocity, double& mu)
		: Function<dim>(dim + 1)
		, velocity(InitialVelocity)
		, mu(mu)
	{}

	template <int dim>
	void
		InitialSolution<dim>::vector_value(const Point<dim>& /*p*/,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + 1), ExcDimensionMismatch(values.size(), dim + 1));
		values[0] = 0;//- velocity/2.0 * M_PI * std::sin(M_PI / 2.0 * p[0]);
		values[1] = 0;// velocity* M_PI* std::sin(M_PI * p[0])* std::sin(M_PI * p[1]);
		values[dim] = 0;
	}
	template <int dim>
	void InitialSolution<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			InitialSolution<dim>::vector_value(points[p], value_list[p]);
	}

	template <int dim>
	class InitialVelocity : public Function<dim>
	{
	public:
		InitialVelocity(double& InitialVelocity, double& mu);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
		const double mu;
	};

	template <int dim>
	InitialVelocity<dim>::InitialVelocity(double& InitialVelocity, double& mu)
		: Function<dim>(dim + 1)
		, velocity(InitialVelocity)
		, mu(mu)
	{}

	template <int dim>
	void
		InitialVelocity<dim>::vector_value(const Point<dim>& /*p*/,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + 1), ExcDimensionMismatch(values.size(), dim + 1));
		values[0] = 0;
		values[1] = 0;
	}
	template <int dim>
	void InitialVelocity<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			InitialVelocity<dim>::vector_value(points[p], value_list[p]);
	}

	template <int dim>
	class InitialAcceleration : public Function<dim>
	{
	public:
		InitialAcceleration(double& InitialVelocity, double& mu, double& dt);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double velocity;
		const double mu;
		const double dt;
	};

	template <int dim>
	InitialAcceleration<dim>::InitialAcceleration(double& InitialVelocity, double& mu, double& dt)
		: Function<dim>(dim + 1)
		, velocity(InitialVelocity)
		, mu(mu)
		, dt(dt)
	{}

	template <int dim>
	void
		InitialAcceleration<dim>::vector_value(const Point<dim>& /*p*/,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + 1), ExcDimensionMismatch(values.size(), dim + 1));
		values[0] = 0;
		values[1] = 0;
	}
	template <int dim>
	void InitialAcceleration<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			InitialAcceleration<dim>::vector_value(points[p], value_list[p]);
	}


	template <int dim>
	class Solution : public Function<dim>
	{
	public:
		Solution(double& present_time, double& velocity, double& kappa);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
		const double a;
		const double kappa;
	};

	template <int dim>
	Solution<dim>::Solution(double& present_time, double& velocity, double& kappa)
		: Function<dim>(dim + 1),
		time(present_time),
		a(velocity),
		kappa(kappa)
	{}

	template <int dim>
	void
		Solution<dim>::vector_value(const Point<dim>& /*p*/,
			Vector<double>& values) const
	{
		//Assert(values.size() == (dim), ExcDimensionMismatch(values.size(), dim));
		values[0] = a * time * time * time/ 6.0;
		values[1] = 0;
		values[2] = 0;
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
	class DirichletValues : public Function<dim>
	{
	public:
		DirichletValues(double& present_time, double& velocity, double& dt, double& mu);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
		const double dt;
		const double a;
		const double mu;
	};

	template <int dim>
	DirichletValues<dim>::DirichletValues(double& present_time, double& velocity, double& dt, double& mu)
		: Function<dim>(dim + 1),
		time(present_time),
		dt(dt),
		a(velocity),
		mu(mu)
	{}

	template <int dim>
	void
		DirichletValues<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + 1), ExcDimensionMismatch(values.size(), dim + 1));
		values[0] = a / 6. * (( (time) *(time) * (time)) - ( (time - dt) * (time - dt) * (time - dt)));
		values[1] = 0;

		//if (abs(p[0] - 1.0) < 0.001) {
		//	values[dim] = -a * dt;// a* dt* std::cos(M_PI * p[0]);
		//}
		//else {
		//	values[dim] = 0;
		//}
	}
	template <int dim>
	void DirichletValues<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			DirichletValues<dim>::vector_value(points[p], value_list[p]);
	}

	template <int dim>
	class Pressure : public Function<dim>
	{
	public:
		Pressure(double& present_time, double& velocity, double& kappa);
		virtual void vector_value(const Point<dim>& p, Vector<double>& value) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
		const double a;
		const double kappa;
	};

	template <int dim>
	Pressure<dim>::Pressure(double& present_time, double& velocity, double& kappa)
		: Function<dim>(1),
		time(present_time),
		a(velocity),
		kappa(kappa)
	{}

	template <int dim>
	void
		Pressure<dim>::vector_value(
			const Point<dim>& p,
			Vector<double>& value) const
	{
		value[0] = -0.25 * a * kappa * M_PI * std::sin(M_PI * time) *
			(-4.0 * std::cos(M_PI * p[1]) * std::sin(M_PI * p[0]) +
				std::cos(M_PI / 2.0 * p[0]) *
				(1.0 + a * M_PI * std::cos(M_PI * p[1]) * std::sin(M_PI * time) * std::sin(M_PI * p[0])));
	}
	template <int dim>
	void Pressure<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Vector<double>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			Pressure<dim>::vector_value(points[p], value_list[p]);
	}


	template <int dim>
	class PressureNeumann : public Function<dim>
	{
	public:
		PressureNeumann(double& present_time, double& velocity, double& kappa);
		virtual void vector_value(const Point<dim>& p,
			Tensor<1, dim>& values) const;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Tensor<1, dim>>& value_list) const;
	private:
		const double time;
		const double a;
		const double kappa;
	};

	template <int dim>
	PressureNeumann<dim>::PressureNeumann(double& present_time, double& velocity, double& kappa)
		: Function<dim>(dim),
		time(present_time),
		a(velocity),
		kappa(kappa)
	{}

	template <int dim>
	void
		PressureNeumann<dim>::vector_value(const Point<dim>& p,
			Tensor<1, dim>& values) const
	{
		Assert(values.n_independent_components == (dim), ExcDimensionMismatch(values.n_independent_components, dim));
		values[0] = 0;

	}
	template <int dim>
	void PressureNeumann<dim>::vector_value_list(
		const std::vector<Point<dim>>& points,
		std::vector<Tensor<1, dim>>& value_list) const
	{
		const unsigned int n_points = points.size();
		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
		for (unsigned int p = 0; p < n_points; ++p)
			PressureNeumann<dim>::vector_value(points[p], value_list[p]);
	}

	template<int dim> // Constructor for the main class
	Incompressible<dim>::Incompressible(const std::string& input_file)
		: parameters(input_file)
		, mapping_simplex(FE_SimplexP_Bubbles<dim>(parameters.velocity_order))
		, dof_handler(triangulation)
		, fe(FE_SimplexP_Bubbles<dim>(parameters.velocity_order), dim, FE_SimplexP<dim>(parameters.pressure_order), 1)
		, quadrature_formula(3)
		, face_quadrature_formula(3)
		, timestep_no(0)
	{}


	//This is a destructor.
	template <int dim>
	Incompressible<dim>::~Incompressible()
	{
		dof_handler.clear();
	}

	// Split up the run function from the grid_generator to replace refinement cycles with timesteps
	template<int dim>
	void Incompressible<dim>::run()
	{
		E = parameters.E;
		nu = parameters.nu;
		mu = get_mu<dim>(E, nu);
		kappa = get_kappa<dim>(E, nu);
		rho_0 = parameters.rho_0;
		present_time = parameters.start_time;
		dt = parameters.dt;
		end_time = parameters.end_time;
		save_time = parameters.save_time;

		create_coarse_grid(triangulation);
		setup_system();


		output_results();
		cout << "Saving results at time : " << present_time << std::endl;
		save_counter = 1;


		assemble_system_Kuu();
		cout << "Lumped mass matrix assembled" << std::endl;

		while (present_time < end_time - 1e-12) {
			do_timestep();
		}
	}

	template <int dim>
	void Incompressible<dim>::create_coarse_grid(Triangulation<2>& triangulation)
	{
		Triangulation<dim> quad_triangulation;

		std::vector<Point<2>> vertices = {
			{0.0,-0.0} , {0.0,1.0}, {1.0,1.0 }, {1.0, 0.0} };

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
		quad_triangulation.create_triangulation(vertices, cells, SubCellData());


		for (const auto& cell : quad_triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (face_center[0] == 0) {
						face->set_boundary_id(1);
					}
					/*if (abs(face_center[0]) < 0.001) {
						face->set_boundary_id(1);
					}*/
				}
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);
		triangulation.refine_global(parameters.n_ref);

	}

	template <int dim>
	void Incompressible<dim>::create_coarse_grid(Triangulation<3>& triangulation)
	{
		Triangulation<dim> quad_triangulation;

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
		quad_triangulation.create_triangulation(vertices, cells, SubCellData());


		/*for (const auto& cell : quad_triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (face_center[0] == 0) {
						face->set_boundary_id(0);
					}
					if (abs(face_center[0] - 1.0) < 0.015) {
						face->set_boundary_id(0);
					}
				}*/
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);

		triangulation.refine_global(parameters.n_ref);
	}







	template <int dim>
	void Incompressible<dim>::setup_system()
	{
		K.clear();

		dof_handler.distribute_dofs(fe);

		std::vector<unsigned int> block_component(dim + 1, 0);
		block_component[dim] = 1;
		DoFRenumbering::component_wise(dof_handler, block_component);



		// HOMOGENEOUS CONSTRAINTS
		{
			constraints.clear();
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(mapping_simplex,
				dof_handler,
				2,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				fe.component_mask(Velocity));
			constraints.close();

		}


		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
		const types::global_dof_index n_u = dofs_per_block[0];
		const types::global_dof_index n_p = dofs_per_block[1];



		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< " (" << n_u << '+' << n_p << ')' << std::endl;





		//DYNAMIC SPARSITY PATTERNS
		{
			BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
			dsp.block(0, 0).reinit(n_u, n_u);
			dsp.block(0, 1).reinit(n_u, n_p);
			dsp.block(1, 0).reinit(n_p, n_u);
			dsp.block(1, 1).reinit(n_p, n_p);
			dsp.collect_sizes();
			DoFTools::make_sparsity_pattern(dof_handler,
				dsp,
				constraints,
				false);
			sparsity_pattern.copy_from(dsp);
		}
		K.reinit(sparsity_pattern);



		{
			BlockDynamicSparsityPattern un_dsp(dofs_per_block, dofs_per_block);
			un_dsp.block(0, 0).reinit(n_u, n_u);
			un_dsp.block(0, 1).reinit(n_u, n_p);
			un_dsp.block(1, 0).reinit(n_p, n_u);
			un_dsp.block(1, 1).reinit(n_p, n_p);
			un_dsp.collect_sizes();
			DoFTools::make_sparsity_pattern(dof_handler, un_dsp);
			un_sparsity_pattern.copy_from(un_dsp);
		}
		un_K.reinit(un_sparsity_pattern);


		R.reinit(dofs_per_block);
		un_R.reinit(dofs_per_block);


		solution.reinit(dofs_per_block);
		solution_increment.reinit(dofs_per_block);

		pressure.reinit(n_p);
		old_pressure.reinit(n_p);



		true_solution.reinit(dofs_per_block);

		error.reinit(dofs_per_block);

		displacement.reinit(n_u);
		old_displacement.reinit(n_u);
		acceleration.reinit(n_u);
		old_acceleration.reinit(n_u);
		velocity.reinit(n_u);
		old_velocity.reinit(n_u);


        u_cell_wise_error.reinit(triangulation.n_active_cells());
		p_cell_wise_error.reinit(triangulation.n_active_cells());

		const FEValuesExtractors::Vector Velocity(0);

		VectorTools::interpolate(mapping_simplex,
			dof_handler,
			InitialVelocity<dim>(parameters.InitialVelocity, mu),
			solution,
			fe.component_mask(Velocity));
		velocity = solution.block(0);

		VectorTools::interpolate(mapping_simplex,
			dof_handler,
			InitialAcceleration<dim>(parameters.BodyForce, mu, dt),
			solution,
			fe.component_mask(Velocity));
		acceleration = solution.block(0);
		VectorTools::interpolate(mapping_simplex, dof_handler, InitialSolution<dim>(parameters.BodyForce, mu), solution);
		pressure_mean = solution.block(1).mean_value();
	}

	template <int dim>
	void Incompressible<dim>::assemble_system_Kuu()
	{

		K = 0;
		un_K = 0;
		R = 0;
		un_R = 0;
		/*const Quadrature<dim> nodal_quad = compute_nodal_quadrature(fe);
		FEValues<dim> fe_lump_values(mapping_simplex,
			fe,
			nodal_quad,
			update_values | 
			update_JxW_values);*/

		FEValues<dim> fe_values(mapping_simplex,
			fe,
			quadrature_formula,
			update_values |
			update_quadrature_points |
			update_JxW_values);
		const std::vector<Point<dim>> q_points = quadrature_formula.get_points();

		/*for (unsigned int i = 0 ; i < quadrature_formula.size() ; ++i)
			cout << q_points[i] << std::endl;*/


		const unsigned int dofs_per_cell = fe.n_dofs_per_cell();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);

		double alpha = parameters.alpha;
		double beta = alpha + 1.0 / 12.0;

		double rho_0 = parameters.rho_0;
		bool lump_mass = parameters.LumpMass;
		if (lump_mass == true) {
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				cell_mass_matrix = 0;
				fe_values.reinit(cell);


				for (const unsigned int q : fe_values.quadrature_point_indices())
				{


					for (const unsigned int i : fe_values.dof_indices())
					{
						double N_p_i = fe_values[Pressure].value(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, j) += -1. / kappa * N_p_i * fe_values[Pressure].value(j, q) * fe_values.JxW(q);

						}
					}
				}
				cell->get_dof_indices(local_dof_indices);
				constraints.distribute_local_to_global(cell_mass_matrix,
					local_dof_indices,
					K);
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						un_K.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
					}
				}
			}
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				cell_mass_matrix = 0;
				fe_values.reinit(cell);


				for (const unsigned int q : fe_values.quadrature_point_indices())
				{


					for (const unsigned int i : fe_values.dof_indices())
					{
						Tensor<1,dim> N_p_i = fe_values[Velocity].value(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, i) += alpha / (beta * dt * dt) * rho_0 * N_p_i * fe_values[Velocity].value(j, q) * fe_values.JxW(q);

						}
					}
				}
				cell->get_dof_indices(local_dof_indices);
				constraints.distribute_local_to_global(cell_mass_matrix,
					local_dof_indices,
					K);
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
						un_K.add(local_dof_indices[i], local_dof_indices[i], cell_mass_matrix(i, i));
				}
			}
		}
		else {
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				cell_mass_matrix = 0;
				fe_values.reinit(cell);
				for (const unsigned int q : fe_values.quadrature_point_indices())
				{
					for (const unsigned int i : fe_values.dof_indices())
					{
						double N_p_i = fe_values[Pressure].value(i, q);
						Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, j) += alpha / (beta * dt * dt) * rho_0 * N_u_i * fe_values[Velocity].value(j, q) * fe_values.JxW(q)
								- 1. / kappa * N_p_i * fe_values[Pressure].value(j, q) * fe_values.JxW(q);
						}
					}
				}
				cell->get_dof_indices(local_dof_indices);
				constraints.distribute_local_to_global(cell_mass_matrix,
					local_dof_indices,
					K);
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						un_K.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));

					}
				}
			}
		}
		};
	

	template <int dim>
	void Incompressible<dim>::assemble_system_not_Kuu()
	{


		un_K.block(0, 1) = 0;
		un_K.block(1, 0) = 0;
		un_R = 0;
		R = 0;
		K.block(0, 1) = 0;
		K.block(1, 0) = 0;



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


		const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
		const unsigned int n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);

		double alpha = parameters.alpha;
		double beta = alpha + 1.0 / 12.0;

		double rho_0 = parameters.rho_0;

		Vector<double>     cell_rhs(dofs_per_cell);

		FExt<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		TractionVector<dim> traction_vector;
		std::vector<Tensor<1, dim>> traction_values(n_face_q_points, Tensor<1, dim>());


		Tensor<2, dim> FF;
		Tensor<2, dim> HH;
		double Jf;
		Tensor<2, dim> pk1;

		double temp_pressure;

		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> face_displacement_grads(n_face_q_points, Tensor<2, dim>());
		std::vector<double> sol_vec_pressure(quadrature_formula.size());
		std::vector<double> face_sol_vec_pressure(n_face_q_points);

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			cell_rhs = 0;
			cell_mass_matrix = 0;
			fe_values.reinit(cell);

			fe_values[Velocity].get_function_gradients(solution, displacement_grads);
			fe_values[Pressure].get_function_values(solution, sol_vec_pressure);

			present_time -= dt;  // Evaluate FExt at time t-1
			right_hand_side.rhs_vector_value_list(fe_values.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time, mu, kappa);
			present_time += dt;


			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				temp_pressure = sol_vec_pressure[q];
				FF = get_real_FF(displacement_grads[q]);
				Jf = get_Jf(FF);
				HH = get_HH(FF, Jf);
				pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);
				//temp_pressure -= pressure_mean;
				for (const unsigned int i : fe_values.dof_indices())
				{
					Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
					double N_p_i = fe_values[Pressure].value(i, q);
					for (const unsigned int j : fe_values.dof_indices())
					{
						cell_mass_matrix(i, j) += (scalar_product(fe_values[Velocity].gradient(i, q), (HH) * fe_values[Pressure].value(j, q)) + //Kup
							N_p_i * Jf * scalar_product(HH, fe_values[Velocity].gradient(j, q)) )* fe_values.JxW(q);

					}
					cell_rhs(i) += (-scalar_product(fe_values[Velocity].gradient(i, q), pk1) +
						rho_0 * fe_values[Velocity].value(i, q) * rhs_values[q] +
						-fe_values[Pressure].value(i, q) * (Jf - 1.0 - temp_pressure / kappa)) * fe_values.JxW(q);

					//std::cout << cell_rhs(i) <<  std::endl;
				}
			}
			for (const auto& face : cell->face_iterators())
			{
				if (face->at_boundary())
				{
					fe_face_values.reinit(cell, face);
					fe_face_values[Velocity].get_function_gradients(solution, face_displacement_grads);
					fe_values[Pressure].get_function_values(solution, face_sol_vec_pressure);



					present_time -= dt;
					traction_vector.traction_vector_value_list(fe_face_values.get_quadrature_points(), traction_values, parameters.TractionMagnitude, present_time);
					present_time += dt;

					for (const unsigned int q : fe_face_values.quadrature_point_indices())
					{
						temp_pressure = face_sol_vec_pressure[q];
						FF = get_real_FF(face_displacement_grads[q]);
						Jf = get_Jf(FF);
						HH = get_HH(FF, Jf);
						pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);
						for (const unsigned int i : fe_face_values.dof_indices())
						{
							if (face->boundary_id() == 2) {
								cell_rhs(i) += fe_face_values[Velocity].value(i, q) * (pk1 * fe_face_values.normal_vector(q)) * fe_face_values.JxW(q);

							}
							if (face->boundary_id() == 1) {
								cell_rhs(i) += fe_face_values[Velocity].value(i, q) * traction_values[q] * fe_face_values.JxW(q);

							}
						}
					}
				}

			}
			//cout << cell_rhs<< std::endl;



			cell->get_dof_indices(local_dof_indices);
			constraints.distribute_local_to_global(cell_mass_matrix,
				cell_rhs,
				local_dof_indices,
				K,
				R);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					un_K.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
				}
				un_R(local_dof_indices[i]) += cell_rhs(i);
			}

		}
	}


	template <int dim>
	void Incompressible<dim>::assemble_Rp()
	{


		un_R.block(1) = 0;
		R.block(1) = 0;

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


		const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
		const unsigned int n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);

		double alpha = parameters.alpha;
		double beta = alpha + 1.0 / 12.0;

		double rho_0 = parameters.rho_0;

		Vector<double>     cell_rhs(dofs_per_cell);

		FExt<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		TractionVector<dim> traction_vector;
		std::vector<Tensor<1, dim>> traction_values(n_face_q_points, Tensor<1, dim>());


		Tensor<2, dim> FF;
		Tensor<2, dim> HH;
		double Jf;
		Tensor<2, dim> pk1;

		double temp_pressure;

		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> face_displacement_grads(n_face_q_points, Tensor<2, dim>());
		std::vector<double> sol_vec_pressure(quadrature_formula.size());
		std::vector<double> face_sol_vec_pressure(n_face_q_points);

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values[Velocity].get_function_gradients(solution, displacement_grads);
			fe_values[Pressure].get_function_values(solution, sol_vec_pressure);


			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				temp_pressure = sol_vec_pressure[q];
				FF = get_real_FF(displacement_grads[q]);
				Jf = get_Jf(FF);
				HH = get_HH(FF, Jf);
				pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);
				//temp_pressure -= pressure_mean;
				for (const unsigned int i : fe_values.dof_indices())
				{

					cell_rhs(i) += (-fe_values[Pressure].value(i, q) * (Jf - 1.0 - temp_pressure / kappa)) * fe_values.JxW(q);

					//std::cout << cell_rhs(i) <<  std::endl;
				}
			}
			//cout << cell_rhs<< std::endl;



			cell->get_dof_indices(local_dof_indices);
			constraints.distribute_local_to_global(cell_rhs,
				local_dof_indices,
				R);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				un_R(local_dof_indices[i]) += cell_rhs(i);
			}

		}

	}





	template<int dim>
	void Incompressible<dim>::solve_semi_implicit()
	{
		{
			constraints.clear();
			//present_time -= dt;
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(mapping_simplex,
				dof_handler,
				2,
				DirichletValues<dim>(present_time, parameters.TractionMagnitude, dt, mu),
				constraints,
				fe.component_mask(Velocity));
			//present_time += dt;

		}
		constraints.close();
		assemble_system_not_Kuu();

		solve_SI_system();
		update_motion();
		calculate_error();

	}

	template<int dim>
	void Incompressible<dim>::solve_explicit()
	{
		{
			constraints.clear();
			//present_time -= dt;
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(mapping_simplex,
				dof_handler,
				2,
				DirichletValues<dim>(present_time, parameters.TractionMagnitude, dt, mu),
				constraints,
				fe.component_mask(Velocity));
			//present_time += dt;

		}
		constraints.close();
		assemble_system_not_Kuu();

		solve_FE_system();
		calculate_error();

	}



	template <int dim>
	void Incompressible<dim>::solve_SI_system()
	{


		const auto& un_Kuu = un_K.block(0, 0);
		const auto& un_Kup = un_K.block(0, 1);
		const auto& un_Kpu = un_K.block(1, 0);
		const auto& un_Kpp = un_K.block(1, 1);


		const auto& Kuu = K.block(0, 0);
		const auto& Kup = K.block(0, 1);
		const auto& Kpu = K.block(1, 0);
		const auto& Kpp = K.block(1, 1);

		const auto op_un_Kuu = linear_operator(un_Kuu);
		const auto op_un_Kup = linear_operator(un_Kup);
		const auto op_un_Kpu = linear_operator(un_Kpu);
		const auto op_un_Kpp = linear_operator(un_Kpp);


		const auto op_Kuu = linear_operator(Kuu);
		const auto op_Kup = linear_operator(Kup);
		const auto op_Kpu = linear_operator(Kpu);
		const auto op_Kpp = linear_operator(Kpp);

		auto& un_Ru = un_R.block(0);
		const auto& un_Rp = un_R.block(1);
		auto& Ru = R.block(0);
		const auto& Rp = R.block(1);

		auto& delta_u = solution_increment.block(0);
		auto& delta_p = solution_increment.block(1);

		SolverControl reduction_control_Kuu(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kuu(reduction_control_Kuu);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kuu;
		PreconditionJacobi<SparseMatrix<double>> preconditioner_un_Kuu;
		preconditioner_Kuu.initialize(Kuu);
		preconditioner_un_Kuu.initialize(un_Kuu);


		SolverControl solver_control_S(30000, 1.0e-12);

		const auto op_Kuu_inv = inverse_operator(op_Kuu, solver_Kuu, preconditioner_Kuu);
		const auto op_un_Kuu_inv = inverse_operator(op_un_Kuu, solver_Kuu, preconditioner_un_Kuu);
		auto op_S = op_Kpp - op_Kpu * op_Kuu_inv * op_Kup;
		auto op_un_S = op_un_Kpp - op_un_Kpu * op_un_Kuu_inv * op_un_Kup;
		if (parameters.nu == 0.5) {
			op_S = -1.0 * op_Kpu * op_Kuu_inv * op_Kup;
			op_un_S = -1.0 * op_un_Kpu * op_un_Kuu_inv * op_un_Kup;
		}

		SolverMinRes<Vector<double>> solver_S(solver_control_S);

		PreconditionIdentity preconditioner_S;

		const double alpha = parameters.alpha;
		const double beta = alpha + 1.0 / 12.0;

		Vector<double> un_motion(acceleration.size());
		un_motion.add((1.0 - alpha / (2.0 * beta)), acceleration, -1.0 * alpha / (beta * dt), velocity);
		double scale = -beta * dt * dt / alpha;
		un_motion *= scale;
		Kuu.vmult_add(Ru, un_motion);

		//Solve for the pressure increment via shur complement
		const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
		const auto op_un_S_inv = inverse_operator(op_un_S, solver_S, preconditioner_S);
		delta_p =  op_S_inv* (Rp - op_Kpu * op_Kuu_inv * Ru);
		constraints.distribute(solution_increment);

		//Solve for velocity
		Ru -= op_Kup * delta_p;
		delta_u = op_Kuu_inv * (Ru);
		constraints.distribute(solution_increment);

	}

	template <int dim>
	void Incompressible<dim>::solve_FE_system()
	{



		const auto& Kuu = K.block(0, 0);
		const auto& Kpp = K.block(1, 1);


		const auto op_Kuu = linear_operator(Kuu);
		const auto op_Kpp = linear_operator(Kpp);

		const auto& Ru = R.block(0);
		const auto& Rp = R.block(1);

		SolverControl reduction_control_Kuu(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kuu(reduction_control_Kuu);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kuu;
		preconditioner_Kuu.initialize(Kuu);

		SolverControl reduction_control_Kpp(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kpp(reduction_control_Kpp);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kpp;
		preconditioner_Kpp.initialize(Kpp);



		const auto op_Kuu_inv = inverse_operator(op_Kuu, solver_Kuu, preconditioner_Kuu);
		const auto op_Kpp_inv = inverse_operator(op_Kpp, solver_Kpp, preconditioner_Kpp);

		

		const double alpha = parameters.alpha;
		const double beta = alpha + 1.0 / 12.0;
		const double gamma = alpha + 0.5;
		
		acceleration = alpha / ( beta * dt * dt ) * op_Kuu_inv * Ru;
		solution.block(0) = solution.block(0) + dt * old_velocity +  dt * dt * ((0.5-beta) * old_acceleration+beta * acceleration);
		velocity = old_velocity + dt * ((1 - gamma) * old_acceleration + gamma * acceleration);
		assemble_Rp();
		solution.block(1) = op_Kpp_inv * Rp;
		//Solve for the pressure increment via shur complement
		

		//Solve for velocity
		
		constraints.distribute(solution);

		old_velocity = velocity;
		old_acceleration = acceleration;

	}

	template<int dim>
	void Incompressible<dim>::update_motion()
	{
		double alpha = parameters.alpha;
		double beta = alpha + 1.0 / 12.0;
		double gamma = 0.5 + alpha;
		swap(velocity, old_velocity);
		swap(acceleration, old_acceleration);

		//std::cout << solution.block(1) << std::endl;
		solution += solution_increment;
		pressure_mean = solution.block(1).mean_value(); //Subtract off average of pressure
		//solution.block(1).add(-mean);
		acceleration = (solution_increment.block(0) - dt * old_velocity + dt * dt * (beta - 0.5) * old_acceleration);
		acceleration *= (1.0 / (dt * dt * beta));

		velocity = old_velocity + dt * ((1.0 - gamma) * old_acceleration + gamma * acceleration);
	}






	template<int dim>
	void Incompressible<dim>::calculate_error()
	{

		//VectorTools::interpolate(dof_handler, Solution<dim>(present_time, parameters.TractionMagnitude, kappa), true_solution);
		//error = (true_solution - solution);
		const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
		const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);

		const FEValuesExtractors::Scalar Pressure(dim);
		const FEValuesExtractors::Vector Velocity(0);

		Solution<dim>(present_time, parameters.TractionMagnitude, kappa);


		QTrapezoid<1>  q_trapez;
		QIterated<dim> quadrature(q_trapez, 5);

		VectorTools::integrate_difference(mapping_simplex,
			dof_handler,
			solution,
			Solution<dim>(present_time, parameters.TractionMagnitude, kappa),
			u_cell_wise_error,
			quadrature_formula,
			VectorTools::L2_norm,
			&velocity_mask);

		displacement_error_output = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::L2_norm);



		VectorTools::integrate_difference(mapping_simplex,
			dof_handler,
			solution,
			Solution<dim>(present_time, parameters.TractionMagnitude, kappa),
			p_cell_wise_error,
			quadrature_formula,
			VectorTools::L2_norm,
			&pressure_mask);

		pressure_error_output = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::L2_norm);

		//cout << "Max displacement error value : " << displacement_error_output << std::endl;

		//cout << "Max pressure error value : " << pressure_error_output << std::endl;
		//present_time += dt;

	}

	//Spits out solution into vectors then into .vtks
	template<int dim>
	void Incompressible<dim>::output_results() const
	{
		/*double mean = mod_solution.block(1).mean_value();
		mod_solution.block(1).add(-mean);*/
		std::vector<std::string> solution_names(dim, "Displacement");
		solution_names.emplace_back("Pressure");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation(dim,
				DataComponentInterpretation::component_is_part_of_vector);
		interpretation.push_back(DataComponentInterpretation::component_is_scalar);


		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(dof_handler,
			solution,
			solution_names,
			interpretation);

		
		data_out.add_data_vector(u_cell_wise_error,
			"Displacement_error",
			DataOut<dim>::type_cell_data);
		data_out.add_data_vector(p_cell_wise_error,
			"Pressure_error",
			DataOut<dim>::type_cell_data);

		BlockVector<double> extra_vector = solution;
		extra_vector.block(0) = velocity;

		std::vector<std::string> extra_names(dim, "Velocity");
		extra_names.emplace_back("Pressure_error2");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation3(dim,
				DataComponentInterpretation::component_is_part_of_vector);
		interpretation3.push_back(DataComponentInterpretation::component_is_scalar);

		data_out.add_data_vector(dof_handler,
			extra_vector,
			extra_names,
			interpretation3);

		BlockVector<double> extra_vector2 = solution;
		extra_vector2.block(0) = acceleration;

		std::vector<std::string> extra_names2(dim, "Acceleration");
		extra_names2.emplace_back("Pressure_error3");
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation4(dim,
				DataComponentInterpretation::component_is_part_of_vector);
		interpretation4.push_back(DataComponentInterpretation::component_is_scalar);

		data_out.add_data_vector(dof_handler,
			extra_vector2,
			extra_names2,
			interpretation4);

		data_out.build_patches(1);
		std::ofstream output("output-" + std::to_string(timestep_no) + ".vtu");
		data_out.write_vtu(output);


	}






	template <int dim>
	void Incompressible<dim>::do_timestep()
	{
		++timestep_no;
		present_time += dt;

		cout << "_____________________________________________________________" << std::endl;
		cout << "Timestep " << timestep_no << " at time " << present_time
			<< std::endl;

		if (parameters.integrator == 0) {
			solve_explicit();
		}
		if (parameters.integrator == 1)
		{
			solve_semi_implicit();
		}


		if (present_time > end_time)
		{
			dt -= (present_time - end_time);
			present_time = end_time;
		}
		if (abs(present_time - save_counter * save_time) < 0.1 * dt) {
			cout << "Saving results at time : " << present_time << std::endl;

			//calculate_error(total_displacement, present_time, displacement_error_output, velocity_error_output);
			output_results();
			save_counter++;
		}
	}


}






//Establsihes namespace, calls PETSc, calls run function, and all is bueno
int main(int /*argc*/, char** /*argv*/)
{
	try
	{
		using namespace dealii;
		using namespace NonlinearElasticity;

		NonlinearElasticity::Incompressible<2> incompressible("parameter_file.prm");
		incompressible.run();
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

