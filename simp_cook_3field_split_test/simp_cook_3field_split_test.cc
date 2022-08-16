#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
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

//For simplices
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

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
				present_timestep = prm.get_double("Timestep");
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
			double tau_FFp;
			double tau_pJ;
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
				prm.declare_entry("Tau_FFp",
					"0.01",
					Patterns::Double(),
					"Tau_FFp");
				prm.declare_entry("Tau_pJ",
					"0.01",
					Patterns::Double(),
					"Tau_pJ");

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
				tau_FFp = prm.get_double("Tau_FFp");
				tau_pJ = prm.get_double("Tau_pJ");
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
		Tensor<2, dim> HH_store;
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
		//Tensor<2, dim> I = unit_symmetric_tensor<dim>();
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
		full_pk1_stress = full_FF;
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

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
		full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3 * full_HH / Jf) + kappa * ((Jf - 1) * full_HH);
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
		Tensor<2, dim> HHF = get_HH(FF, Jf);
		Tensor<2, dim> pk1 = get_pk1(FF, mu, Jf, HHF);

		return pk1;
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
		void         assemble_momentum_mass();
		void		 assemble_pressure_mass();
		void		 assemble_def_grad_mass();
        void         assemble_pressure_Lap(Vector<double>& sol_n_def_grad);
		void		 assemble_def_grad_rhs(Vector<double>& sol_n_momentum);
		void		 assemble_momentum_int_rhs(Vector<double>& sol_n_def_grad, Vector<double>& sol_n_pressure);
		void		 assemble_pressure_rhs(Vector<double>& sol_n_plus_1_momentum, Vector<double>& sol_n_plus_1_def_grad);
		void		 assemble_momentum_rhs(Vector<double>& sol_n_pressure, Vector<double>& sol_n_plus_1_pressure);
		void         solve_ForwardEuler();
		void         solve_ssprk2();
//		void         solve_ssprk3();
		void		 solve_momentum_int(Vector<double>& sol_n, Vector<double>& sol_n_plus_1);
		void		solve_F(Vector<double>& sol_n, Vector<double>& sol_n_plus_1);
		void		solve_p(Vector<double>& sol_n, Vector<double>& sol_n_plus_1);
		void		solve_momentum(Vector<double>& sol_n_momentum, Vector<double>& sol_n_plus_1_momentum);
		void         output_results(Vector<double>& momentum_solution, Vector<double>& pressure_solution, Vector<double>& def_grad_solution) const;

		void do_timestep();

		void update_displacement(const Vector<double>& sol_n, const double& coeff_n, const Vector<double>& sol_n_plus, const double& coeff_n_plus);


		void setup_quadrature_point_history();

		Parameters::AllParameters parameters;

		Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler_momentum;
		DoFHandler<dim>	   dof_handler_def_grad;
		DoFHandler<dim>    dof_handler_pressure;
		FESystem<dim> fe_momentum;
		FESystem<dim> fe_pressure;
		FESystem<dim> fe_def_grad;


		MappingFE<dim> mapping_simplex;

		AffineConstraints<double> homogeneous_constraints_momentum;
        AffineConstraints<double> homogeneous_constraints_def_grad;
        AffineConstraints<double> homogeneous_constraints_pressure;

		const QGaussSimplex<dim> quadrature_formula_momentum;
		const QGaussSimplex<dim - 1> face_quadrature_formula_momentum;
		const QGaussSimplex<dim> quadrature_formula_def_grad;
		const QGaussSimplex<dim - 1> face_quadrature_formula_def_grad;
		const QGaussSimplex<dim> quadrature_formula_pressure;
		const QGaussSimplex<dim - 1> face_quadrature_formula_pressure;

		std::vector<PointHistory<dim>> quadrature_point_history;

		SparsityPattern constrained_sparsity_pattern_momentum;
		SparsityPattern unconstrained_sparsity_pattern_momentum;
		SparseMatrix<double> constrained_mass_matrix_momentum;
		SparseMatrix<double> unconstrained_mass_matrix_momentum;

		SparsityPattern constrained_sparsity_pattern_pressure;
		SparsityPattern unconstrained_sparsity_pattern_pressure;
		SparseMatrix<double> constrained_mass_matrix_pressure;
		SparseMatrix<double> unconstrained_mass_matrix_pressure;

		SparseMatrix<double> constrained_Lap_matrix_pressure;
		SparseMatrix<double> unconstrained_Lap_matrix_pressure;
        
        SparseMatrix<double> constrained_it_matrix_pressure;
        SparseMatrix<double> unconstrained_it_matrix_pressure;

		SparsityPattern constrained_sparsity_pattern_def_grad;
		SparsityPattern unconstrained_sparsity_pattern_def_grad;
		SparseMatrix<double> constrained_mass_matrix_def_grad;
		SparseMatrix<double> unconstrained_mass_matrix_def_grad;


		Vector<double> momentum_rhs;
		Vector<double> def_grad_rhs;
		Vector<double> pressure_rhs;


		Vector<double> momentum_solution;
		Vector<double> momentum_old_solution;
		Vector<double> momentum_int_solution;   //For RK2 and higher order
		Vector<double> momentum_int_solution_2; //For RK3 and higher order

		Vector<double> def_grad_solution;
		Vector<double> def_grad_old_solution;
		Vector<double> def_grad_int_solution;   //For RK2 and higher order
		Vector<double> def_grad_int_solution_2; //For RK3 and higher order

		Vector<double> pressure_solution;
		Vector<double> pressure_old_solution;
		Vector<double> pressure_int_solution;   //For RK2 and higher order
		Vector<double> pressure_int_solution_2; //For RK3 and higher order

		//Vector<double> residual;

		Vector<double> incremental_displacement;
		Vector<double> total_displacement;


		double present_time;
		double present_timestep;
		double end_time;
		double save_time;
		double save_counter;
		unsigned int timestep_no;


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
			DataPostprocessorTensor<dim>("FF_real",
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

	template <int dim>
	class DisplacementPostprocessor : public DataPostprocessorVector<dim>
	{
	public:
		DisplacementPostprocessor()
			:
			// call the constructor of the base class. call the variable to
			// be output "grad_u" and make sure that DataOut provides us
			// with the gradients:
			DataPostprocessorVector<dim>("Displacement",
				update_values)
		{}

		virtual
			void
			evaluate_vector_field
			(const DataPostprocessorInputs::Vector<dim>& input_data,
				std::vector<Vector<double> >& computed_quantities) const override
		{
			// ensure that there really are as many output slots
			// as there are points at which DataOut provides the
			// gradients:
			AssertDimension(computed_quantities.size(), input_data.solution_values.size());

			// then loop over all of these inputs:
			for (unsigned int p = 0; p < input_data.solution_values.size(); ++p)
			{
				for (unsigned int d = 0; d < dim; ++d)
					computed_quantities[p][d] = input_data.solution_values[p][d];
			}
		}
	};

	template <int dim>
	class PressurePostprocessor : public DataPostprocessorScalar<dim>
	{
	public:
		PressurePostprocessor()
			:
			// call the constructor of the base class. call the variable to
			// be output "grad_u" and make sure that DataOut provides us
			// with the gradients:
			DataPostprocessorScalar<dim>("Real_Pressure",
				update_gradients)
		{}

		virtual
			void
			evaluate_vector_field
			(const DataPostprocessorInputs::Vector<dim>& input_data,
				std::vector<Vector<double>>& computed_quantities) const override
		{
			// ensure that there really are as many output slots
			// as there are points at which DataOut provides the
			// gradients:
			AssertDimension(computed_quantities.size(), input_data.solution_values.size());
			Tensor<2, dim> FF;
			Tensor<2, dim> I = unit_symmetric_tensor<dim>();

			// then loop over all of these inputs:
			for (unsigned int p = 0; p < input_data.solution_values.size(); ++p)
			{
				FF = 0;
				for (unsigned int d = 0; d < dim; ++d)
					for (unsigned int e = 0; e < dim; ++e)
						FF[d][e] = I[d][e] + input_data.solution_gradients[p][d][e];

				computed_quantities[p] = (determinant(FF) - 1);
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
	class Def_Grad_bound : public Function<dim>
	{
	public:
		Def_Grad_bound() : Function<dim>(dim * dim)
		{}
		void
			vector_value(const Point<dim>& /*p*/,
				Vector<double>& values) const override
		{
			values = 0;
		}
	};

	template <int dim>
	class Boundary_Shear : public Function<dim>
	{
	public:
		Boundary_Shear() : Function<dim>(dim + 1 + dim * dim)
		{}
		void
			vector_value(const Point<dim>& /*p*/,
				Vector<double>& values) const override
		{
			values = 0;
			values(1) = 2;
		}
	};



	template<int dim> // Constructor for the main class
	Incompressible<dim>::Incompressible(const std::string& input_file)
		: parameters(input_file)
		, mapping_simplex(FE_SimplexP<dim>(parameters.order+1))
		, dof_handler_momentum(triangulation)
		, dof_handler_pressure(triangulation)
		, dof_handler_def_grad(triangulation)
		, fe_momentum(FE_SimplexP<dim>(parameters.order+1),dim)
		, fe_pressure(FE_SimplexP<dim>(parameters.order),1)
		, fe_def_grad(FE_SimplexP<dim>(parameters.order+1),dim*dim)
		, quadrature_formula_momentum(parameters.order + 3)
		, quadrature_formula_pressure(parameters.order + 3)
		, quadrature_formula_def_grad(parameters.order + 3)
		, face_quadrature_formula_momentum(parameters.order + 3)
		, face_quadrature_formula_pressure(parameters.order + 2)
		, face_quadrature_formula_def_grad(parameters.order + 1)
		, timestep_no(0)
	{}


	//This is a destructor.
	template <int dim>
	Incompressible<dim>::~Incompressible()
	{
		dof_handler_momentum.clear();
        dof_handler_pressure.clear();
        dof_handler_def_grad.clear();

	}

	// Split up the run function from the grid_generator to replace refinement cycles with timesteps
	template<int dim>
	void Incompressible<dim>::run()
	{
		create_coarse_grid(triangulation);
		setup_system();
		E = parameters.E;
		nu = parameters.nu;
		present_time = parameters.start_time;
		present_timestep = parameters.present_timestep;
		end_time = parameters.end_time;
		save_time = parameters.save_time;
		mu = get_mu<dim>(E, nu);
		kappa = get_kappa<dim>(E, nu);
		output_results(momentum_old_solution, pressure_old_solution, def_grad_old_solution);
		cout << "Saving results at time : " << present_time << std::endl;
		save_counter = 1;
        
        assemble_momentum_mass();

        assemble_def_grad_mass();
        assemble_pressure_mass();

		cout << " Mass matrices assembled" << std::endl;

        while (present_time < end_time){
            unconstrained_it_matrix_pressure = 0;
            constrained_it_matrix_pressure = 0;

            assemble_pressure_Lap(def_grad_old_solution);
			cout << "Lap matrix assembled" << std::endl;
			unconstrained_it_matrix_pressure.copy_from(unconstrained_mass_matrix_pressure);
			unconstrained_it_matrix_pressure.add(1.0, unconstrained_Lap_matrix_pressure);	
			constrained_it_matrix_pressure.copy_from(constrained_mass_matrix_pressure);
			constrained_it_matrix_pressure.add(1.0, constrained_Lap_matrix_pressure);
			do_timestep();
        }
	}

	template <int dim>
	void Incompressible<dim>::create_coarse_grid(Triangulation<2>& triangulation)
	{
		Triangulation<dim> quad_triangulation;

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
		quad_triangulation.create_triangulation(vertices, cells, SubCellData());


		for (const auto& cell : quad_triangulation.active_cell_iterators())
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
		cout << quad_triangulation.n_global_levels() << std::endl;
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);
		triangulation.refine_global(parameters.n_ref);

		setup_quadrature_point_history();
	}

	template <int dim>
	void Incompressible<dim>::create_coarse_grid(Triangulation<3>& triangulation)
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
	void Incompressible<dim>::setup_system()
	{

		dof_handler_momentum.distribute_dofs(fe_momentum);
        dof_handler_def_grad.distribute_dofs(fe_def_grad);
        dof_handler_pressure.distribute_dofs(fe_pressure);


		std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
        << "Number of degrees of freedom: " << dof_handler_momentum.n_dofs() + dof_handler_def_grad.n_dofs() +dof_handler_pressure.n_dofs()
			<< " (" << dof_handler_momentum.n_dofs() << '+' << dof_handler_pressure.n_dofs() << '+' << dof_handler_def_grad.n_dofs() << ')' << std::endl;

		std::cout << "Setting up zero boundary conditions" << std::endl;

		FEValuesExtractors::Vector Momentum(0);
		
        

// HOMOGENEOUS CONSTRAINTS
        homogeneous_constraints_momentum.clear();
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
			dof_handler_momentum,
			1,
			Functions::ZeroFunction<dim>(dim),
			homogeneous_constraints_momentum,
            fe_momentum.component_mask(Momentum) );
        homogeneous_constraints_momentum.close();

        homogeneous_constraints_pressure.clear();
		//dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
		//	dof_handler,
		//	4,
		//	Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
		//	homogeneous_constraints_pressure);
        homogeneous_constraints_pressure.close();
        std::cout << "Boundary conditions established" << std::endl;

        homogeneous_constraints_def_grad.clear();
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
			dof_handler_def_grad,
			1,
			Functions::ZeroFunction<dim>(dim*dim),
            homogeneous_constraints_def_grad);
        homogeneous_constraints_def_grad.close();


//DYNAMIC SPARSITY PATTERNS
		DynamicSparsityPattern dsp_momentum_constrained(dof_handler_momentum.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler_momentum,
			dsp_momentum_constrained,
			homogeneous_constraints_momentum,
			false);
		constrained_sparsity_pattern_momentum.copy_from(dsp_momentum_constrained);
		constrained_mass_matrix_momentum.reinit(constrained_sparsity_pattern_momentum);

		DynamicSparsityPattern dsp_momentum_unconstrained(dof_handler_momentum.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler_momentum, dsp_momentum_unconstrained);
		unconstrained_sparsity_pattern_momentum.copy_from(dsp_momentum_unconstrained);
		unconstrained_mass_matrix_momentum.reinit(unconstrained_sparsity_pattern_momentum);

        
        DynamicSparsityPattern dsp_pressure_constrained(dof_handler_pressure.n_dofs(), dof_handler_pressure.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_pressure,
            dsp_pressure_constrained,
            homogeneous_constraints_pressure,
            false);
        constrained_sparsity_pattern_pressure.copy_from(dsp_pressure_constrained);
        constrained_mass_matrix_pressure.reinit(constrained_sparsity_pattern_pressure);

        DynamicSparsityPattern dsp_pressure_unconstrained(dof_handler_pressure.n_dofs(), dof_handler_pressure.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp_pressure_unconstrained);
        unconstrained_sparsity_pattern_pressure.copy_from(dsp_pressure_unconstrained);
        unconstrained_mass_matrix_pressure.reinit(unconstrained_sparsity_pattern_pressure);

		constrained_Lap_matrix_pressure.reinit(constrained_sparsity_pattern_pressure);
		unconstrained_Lap_matrix_pressure.reinit(unconstrained_sparsity_pattern_pressure);

        constrained_it_matrix_pressure.reinit(constrained_sparsity_pattern_pressure);
        unconstrained_it_matrix_pressure.reinit(unconstrained_sparsity_pattern_pressure);
        
        DynamicSparsityPattern dsp_def_grad_constrained(dof_handler_def_grad.n_dofs(), dof_handler_def_grad.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_def_grad,
            dsp_def_grad_constrained,
            homogeneous_constraints_def_grad,
            false);
        constrained_sparsity_pattern_def_grad.copy_from(dsp_def_grad_constrained);
        constrained_mass_matrix_def_grad.reinit(constrained_sparsity_pattern_def_grad);

        DynamicSparsityPattern dsp_def_grad_unconstrained(dof_handler_def_grad.n_dofs(), dof_handler_def_grad.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_def_grad, dsp_def_grad_unconstrained);
        unconstrained_sparsity_pattern_def_grad.copy_from(dsp_def_grad_unconstrained);
        unconstrained_mass_matrix_def_grad.reinit(unconstrained_sparsity_pattern_def_grad);

		momentum_solution.reinit(dof_handler_momentum.n_dofs());
		momentum_old_solution.reinit(dof_handler_momentum.n_dofs());
		momentum_int_solution.reinit(dof_handler_momentum.n_dofs());
		momentum_int_solution_2.reinit(dof_handler_momentum.n_dofs());
		momentum_rhs.reinit(dof_handler_momentum.n_dofs());
        
        def_grad_solution.reinit(dof_handler_def_grad.n_dofs());
        def_grad_old_solution.reinit(dof_handler_def_grad.n_dofs());
        def_grad_int_solution.reinit(dof_handler_def_grad.n_dofs());
        def_grad_int_solution_2.reinit(dof_handler_def_grad.n_dofs());
        def_grad_rhs.reinit(dof_handler_def_grad.n_dofs());

        pressure_solution.reinit(dof_handler_pressure.n_dofs());
        pressure_old_solution.reinit(dof_handler_pressure.n_dofs());
        pressure_int_solution.reinit(dof_handler_pressure.n_dofs());
        pressure_int_solution_2.reinit(dof_handler_pressure.n_dofs());
        pressure_rhs.reinit(dof_handler_pressure.n_dofs());


		cout << "Applying initial conditions" << std::endl;
		VectorTools::interpolate(dof_handler_momentum, InitialMomentum<dim>(parameters.InitialVelocity), momentum_old_solution);
        
        VectorTools::interpolate(dof_handler_def_grad, Def_Grad_bound<dim>(), def_grad_old_solution);

		incremental_displacement.reinit(dof_handler_momentum.n_dofs());
        total_displacement.reinit(dof_handler_momentum.n_dofs());
	}

	template <int dim>
	void Incompressible<dim>::assemble_momentum_mass()
	{

        const FEValuesExtractors::Vector Momentum(0);

        constrained_mass_matrix_momentum = 0;

        unconstrained_mass_matrix_momentum = 0;

		FEValues<dim> fe_values(mapping_simplex,
			fe_momentum,
			quadrature_formula_momentum,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = fe_momentum.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula_momentum.size();

		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



		Tensor<1, dim> fe_val_Momentum_i;


		for (const auto& cell : dof_handler_momentum.active_cell_iterators())
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
			homogeneous_constraints_momentum.distribute_local_to_global(
				cell_mass_matrix,
				local_dof_indices,
				constrained_mass_matrix_momentum);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					unconstrained_mass_matrix_momentum.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
				}
			}
		}
	}

template <int dim>
void Incompressible<dim>::assemble_pressure_mass()
{
    FEValuesExtractors::Scalar Pressure(0);


    constrained_mass_matrix_pressure = 0;

    unconstrained_mass_matrix_pressure = 0;

    FEValues<dim> fe_values(mapping_simplex,
        fe_pressure,
        quadrature_formula_pressure,
        update_values |
        update_gradients |
        update_quadrature_points |
        update_JxW_values);


    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula_pressure.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



	double fe_val_Pressure_i;


    for (const auto& cell : dof_handler_pressure.active_cell_iterators())
    {
        

        cell_mass_matrix = 0;
        fe_values.reinit(cell);
        for (const unsigned int q_point : fe_values.quadrature_point_indices())
        {
            for (const unsigned int i : fe_values.dof_indices())
            {
                fe_val_Pressure_i = fe_values[Pressure].value(i, q_point);
                for (const unsigned int j : fe_values.dof_indices())
                {
                    cell_mass_matrix(i, j) += 1.0 / kappa *
                        fe_val_Pressure_i * //Momentum terms
                        fe_values[Pressure].value(j, q_point) *
                    fe_values.JxW(q_point);
                }

            }
        }



        cell->get_dof_indices(local_dof_indices);
        homogeneous_constraints_pressure.distribute_local_to_global(
            cell_mass_matrix,
            local_dof_indices,
            constrained_mass_matrix_pressure);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                unconstrained_mass_matrix_pressure.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
            }
        }
    }
}

template <int dim>
void Incompressible<dim>::assemble_def_grad_mass()
{

	FEValuesExtractors::Tensor<dim> Def_Grad(0);
    
    constrained_mass_matrix_def_grad = 0;

    unconstrained_mass_matrix_def_grad = 0;

    FEValues<dim> fe_values_def_grad(mapping_simplex,
        fe_def_grad,
        quadrature_formula_def_grad,
        update_values |
        update_gradients |
        update_quadrature_points |
        update_JxW_values);


    const unsigned int dofs_per_cell = fe_def_grad.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula_def_grad.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



    Tensor<2, dim> fe_val_Def_Grad_i;


    for (const auto& cell : dof_handler_def_grad.active_cell_iterators())
    {
        

        cell_mass_matrix = 0;
        fe_values_def_grad.reinit(cell);



        for (const unsigned int q_point : fe_values_def_grad.quadrature_point_indices())
        {
            for (const unsigned int i : fe_values_def_grad.dof_indices())
            {
                fe_val_Def_Grad_i = fe_values_def_grad[Def_Grad].value(i, q_point);
                for (const unsigned int j : fe_values_def_grad.dof_indices())
                {
                    cell_mass_matrix(i, j) +=
                       scalar_product( fe_val_Def_Grad_i ,
                        fe_values_def_grad[Def_Grad].value(j, q_point)) *
                    fe_values_def_grad.JxW(q_point);
                }

            }
        }



        cell->get_dof_indices(local_dof_indices);
        homogeneous_constraints_def_grad.distribute_local_to_global(
            cell_mass_matrix,
            local_dof_indices,
            constrained_mass_matrix_def_grad);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                unconstrained_mass_matrix_def_grad.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
            }
        }
    }
}

template <int dim>
void Incompressible<dim>::assemble_pressure_Lap(Vector<double>& sol_n_def_grad)
{

    FEValuesExtractors::Scalar Pressure(0);

	constrained_Lap_matrix_pressure = 0;
	unconstrained_Lap_matrix_pressure = 0;

    FEValues<dim> fe_values_pressure(mapping_simplex,
        fe_pressure,
        quadrature_formula_pressure,
        update_values |
        update_gradients |
        update_quadrature_points |
        update_JxW_values);
    
    FEValues<dim> fe_values_def_grad(mapping_simplex,
                                     fe_def_grad,
                                     quadrature_formula_def_grad,
                                     update_values|
                                     update_gradients|
                                     update_quadrature_points|
                                     update_JxW_values);


    const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula_pressure.size();


    std::vector<Vector<double>> sol_vec_def_grad(n_q_points, Vector<double>(dim * dim));


    FullMatrix<double> cell_Lap_matrix(dofs_per_cell, dofs_per_cell);




    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    Tensor<2, dim> real_FF;
    Tensor<2, dim> FF;
    Tensor<2, dim> HH;
    Tensor<2, dim> real_HH;
    double Jf;
    double sol_counter;
    double real_Jf;
    double real_pressure;


    std::vector<std::vector<Tensor<1, dim>>> displacement_grads(quadrature_formula_momentum.size(),std::vector<Tensor<1, dim>>(dim));

    Tensor<1, dim> fe_grad_Pressure_i;

    //Stability parameters
    double alpha = parameters.alpha;
    double beta = parameters.beta;

    //Tensor<2, dim> II = unit_symmetric_tensor<dim>();
	//using IteratorTuple = std::tuple<typename DoFHandler<dim>::active_cell_iterator,
	//								 typename DoFHandler<dim>::active_cell_iterator>;

	//using IteratorPair = SynchronousIterators<IteratorTuple>;

	//int iterator_counter=0;
	//IteratorPair(IteratorTuple(dof_handler_pressure.begin_active(),
	//	dof_handler_def_grad.begin_active()));
	//IteratorPair(IteratorTuple(dof_handler_pressure.end(),
	//	dof_handler_def_grad.end()));

	auto def_grad_cell = dof_handler_def_grad.begin_active();
	for( auto cell : dof_handler_pressure.active_cell_iterators())
	{
        PointHistory<dim>* local_quadrature_points_history = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
        //real_FF = 0;
        FF = 0;
        HH = 0;
        Jf = 0;
        //temp_pressure = 0;

        //real_Jf = 0;
       // real_pressure = 0;

        cell_Lap_matrix = 0;
        fe_values_pressure.reinit(cell);
        fe_values_def_grad.reinit(def_grad_cell);

        fe_values_def_grad.get_function_values(sol_n_def_grad, sol_vec_def_grad);


        for (const unsigned int q_point : fe_values_pressure.quadrature_point_indices())
        {



            double sol_counter = 0;

            for (unsigned int i = 0; i < dim; i++) {
                for (unsigned int j = 0; j < dim; j++) { // Extracts deformation gradient values, puts them in tensor form
                    FF[i][j] = sol_vec_def_grad[q_point](sol_counter);
                    ++sol_counter;
                }
            }

            //fe_values_def_grad.get_function_gradients(total_displacement, displacement_grads);
            //real_FF = get_real_FF(displacement_grads[q_point]);
            //FF += alpha * (real_FF - FF);
            //real_Jf = get_Jf(real_FF);
            Jf = get_Jf(FF);
            HH = unit_symmetric_tensor<dim>();
            //real_HH = get_HH(real_FF, real_Jf);

            //cout << "q_point momentum : " << temp_momentum << std::endl;
           
			/*cout << "real FF : " << real_FF << std::endl;
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    cout << "displacement_grads[" << i << j << "]: " << displacement_grads[q_point][i][j] << std::endl;*/
                    //cout << "displacement grads : " << displacement_grads[q_point] << std::endl;


            for (const unsigned int i : fe_values_pressure.dof_indices())
            {
                fe_grad_Pressure_i = fe_values_pressure[Pressure].gradient(i, q_point);
                for (const unsigned int j : fe_values_pressure.dof_indices())
                {
                    cell_Lap_matrix(i, j) += present_timestep * present_timestep *
                        scalar_product(HH * fe_grad_Pressure_i,
                            HH * fe_values_pressure[Pressure].gradient(j, q_point)) *
                        fe_values_pressure.JxW(q_point);
                }

            }
        }



        cell->get_dof_indices(local_dof_indices);
        homogeneous_constraints_pressure.distribute_local_to_global(
            cell_Lap_matrix,
            local_dof_indices,
            constrained_Lap_matrix_pressure);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                unconstrained_Lap_matrix_pressure.add(local_dof_indices[i], local_dof_indices[j], cell_Lap_matrix(i, j));
            }
        }
		def_grad_cell++;
	}
}



	template <int dim>
	void Incompressible<dim>::assemble_def_grad_rhs(Vector<double>& sol_n_momentum)
	{
		def_grad_rhs = 0;
		FEValuesExtractors::Tensor<dim> Def_Grad(0);


		FEValues<dim> fe_values_def_grad(mapping_simplex,
			fe_def_grad,
			quadrature_formula_def_grad,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values_def_grad(mapping_simplex,
			fe_def_grad,
			face_quadrature_formula_def_grad,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		FEValues<dim> fe_values_momentum(mapping_simplex,
			fe_momentum,
			quadrature_formula_momentum,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values_momentum(mapping_simplex,
			fe_momentum,
			face_quadrature_formula_momentum,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		const unsigned int dofs_per_cell = fe_def_grad.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula_def_grad.size();
		const unsigned int n_face_q_points = face_quadrature_formula_def_grad.size();

		std::vector<Vector<double>> sol_vec(n_q_points, Vector<double>(dim));
		std::vector<Vector<double>> face_sol_vec(n_face_q_points, Vector<double>(dim));





		Vector<double>     cell_rhs(dofs_per_cell);




		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);






		Tensor<1, dim> temp_momentum;
		Tensor<1, dim> face_temp_momentum;



		auto cell_momentum = dof_handler_momentum.begin_active();
		for (const auto& cell : dof_handler_def_grad.active_cell_iterators())
		{
			
			temp_momentum = 0;
			cell_rhs = 0;
			fe_values_def_grad.reinit(cell);
			fe_values_momentum.reinit(cell_momentum);

			fe_values_momentum.get_function_values(sol_n_momentum, sol_vec);





			for (const unsigned int q_point : fe_values_def_grad.quadrature_point_indices())
			{
				for (unsigned int i = 0; i < dim; i++) { //Extracts momentum values, puts them in vector form

					temp_momentum[i] = sol_vec[q_point](i);

				}

				
				for (unsigned int i=0; i < dofs_per_cell; ++i)
				{
					//cell_rhs(i) += -scalar_product(fe_values[Def_Grad].gradient(i, q_point), outer_product(temp_momentum, II)) * fe_values.JxW(q_point);
					cell_rhs(i) += -fe_values_def_grad[Def_Grad].divergence(i, q_point) * temp_momentum * fe_values_def_grad.JxW(q_point);
				}
			}

			//for (const auto& face : cell->face_iterators())
			//{

			//	face_temp_momentum = 0;
			//	if (face->at_boundary())
			//	{
			//		fe_face_values_def_grad.reinit(cell, face);
			//		fe_face_values_momentum.reinit(cell_momentum, face);
			//		fe_face_values_momentum.get_function_values(sol_n_momentum, face_sol_vec);

			//		for (const unsigned int q_point : fe_face_values_def_grad.quadrature_point_indices())
			//		{
			//			for (int i = 0; i < dim; i++) {
			//				face_temp_momentum[i] = face_sol_vec[q_point](i);
			//			}

			//			for (const unsigned int i : fe_values_def_grad.dof_indices())
			//			{
			//				//if (face->boundary_id() == 5) {
			//				cell_rhs(i) +=   /*scalar_product(fe_face_values[Def_Grad].value(i, q_point), contract<2, 0>(outer_product(temp_momentum, II), fe_face_values.normal_vector(q_point)))*/
			//					fe_face_values_def_grad[Def_Grad].value(i, q_point) * face_temp_momentum * fe_face_values_def_grad.normal_vector(q_point) *
			//					fe_face_values_def_grad.JxW(q_point); // Deformation gradient face terms
			//			//}
			//			//else {
			//			//	cell_rhs(i) +=  scalar_product(fe_face_values[Def_Grad].value(i, q_point), outer_product(temp_momentum,II)*fe_face_values.normal_vector(q_point)) *
			//			//		fe_face_values.JxW(q_point); // Deformation gradient face terms
			//			//}
			//			}
			//		}
			//	}
			//}


			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				def_grad_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
			cell_momentum++;
		}
	}

	template <int dim>
	void Incompressible<dim>::assemble_momentum_int_rhs(Vector<double>& sol_n_def_grad, Vector<double>& sol_n_pressure)
	{
		momentum_rhs = 0;


        FEValues<dim> fe_values_momentum(mapping_simplex,
            fe_momentum,
            quadrature_formula_momentum,
            update_values |
            update_gradients |
            update_quadrature_points |
            update_JxW_values);

        FEFaceValues<dim> fe_face_values_momentum(mapping_simplex,
            fe_momentum,
            face_quadrature_formula_momentum,
            update_values |
            update_gradients |
            update_normal_vectors |
            update_quadrature_points |
            update_JxW_values);

		const unsigned int dofs_per_cell = fe_momentum.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula_momentum.size();
		const unsigned int n_face_q_points = face_quadrature_formula_momentum.size();


        std::vector<std::vector<Tensor<1, dim>>> displacement_grads(quadrature_formula_momentum.size(),std::vector<Tensor<1, dim>>(dim));



		Vector<double>     cell_rhs(dofs_per_cell);

		RightHandSide<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		TractionVector<dim> traction_vector;
		std::vector<Tensor<1, dim>> traction_values(n_face_q_points, Tensor<1, dim>());
        
        const FEValuesExtractors::Vector Momentum(0);


		Tensor<2, dim> FF;
		Tensor<2, dim> pk1;
        double temp_pressure;
		Tensor<2, dim> HH;
		double Jf;


		for (const auto& cell : dof_handler_momentum.active_cell_iterators())
		{
            
            
			FF = 0;
			Jf = 0;
			HH = 0;
			temp_pressure = 0;
			cell_rhs = 0;
			pk1 = 0;
			fe_values_momentum.reinit(cell);

			right_hand_side.rhs_vector_value_list(fe_values_momentum.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time);
            fe_values_momentum.get_function_gradients(total_displacement, displacement_grads);


			for (const unsigned int q_point : fe_values_momentum.quadrature_point_indices())
			{
                
				

				FF = get_real_FF(displacement_grads[q_point]);
				//pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);


				for (unsigned int i : fe_values_momentum.dof_indices())
				{
                    cell_rhs(i) += (-scalar_product(fe_values_momentum[Momentum].gradient(i, q_point), FF) +
                        fe_values_momentum[Momentum].value(i, q_point) * rhs_values[q_point]) * fe_values_momentum.JxW(q_point);
				}
			}

			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				momentum_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
		}
	}
	template <int dim>
	void Incompressible<dim>::assemble_pressure_rhs(Vector<double>& sol_n_plus_1_momentum, Vector<double>& sol_n_plus_1_def_grad)
	{
		pressure_rhs = 0;
        
        FEValuesExtractors::Scalar Pressure(0);

		FEValues<dim> fe_values_momentum(mapping_simplex,
			fe_momentum,
			quadrature_formula_momentum,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);
		FEValues<dim> fe_values_def_grad(mapping_simplex,
			fe_def_grad,
			quadrature_formula_def_grad,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);
		FEValues<dim> fe_values_pressure(mapping_simplex,
			fe_pressure,
			quadrature_formula_pressure,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values_pressure(mapping_simplex,
			fe_pressure,
			face_quadrature_formula_pressure,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values_momentum(mapping_simplex,
			fe_momentum,
			face_quadrature_formula_momentum,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values_def_grad(mapping_simplex,
			fe_def_grad,
			face_quadrature_formula_def_grad,
			update_values |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);

		const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula_pressure.size();
		const unsigned int n_face_q_points = face_quadrature_formula_pressure.size();

		std::vector<Vector<double>> sol_vec_momentum(n_q_points, Vector<double>(dim));
		std::vector<Vector<double>> sol_vec_def_grad(n_q_points, Vector<double>(dim*dim));
		//std::vector<Vector<double>> residual_vec(n_q_points, Vector<double>(dim + 1 + dim * dim));
		std::vector<Vector<double>> face_sol_vec_momentum(n_face_q_points, Vector<double>(dim));
		std::vector<Vector<double>> face_sol_vec_def_grad(n_face_q_points, Vector<double>(dim  * dim));




		Vector<double>     cell_rhs(dofs_per_cell);




		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



		Tensor<2, dim> FF;
		Tensor<2, dim> face_FF;

		Tensor<1, dim> temp_momentum;
		//Tensor<1, dim> temp_momentum_residual;
		Tensor<1, dim> face_temp_momentum;
		Tensor<2, dim> HH;
		double Jf;
		Tensor<2, dim> face_HH;
		int sol_counter;

		double tau_pJ = parameters.tau_pJ * present_timestep;


		auto cell_momentum = dof_handler_momentum.begin_active();
		auto cell_def_grad = dof_handler_def_grad.begin_active();
		for (const auto& cell : dof_handler_pressure.active_cell_iterators())
		{
			FF = 0;
			Jf = 0;
			temp_momentum = 0;
			//temp_momentum_residual = 0;
			cell_rhs = 0;
			fe_values_pressure.reinit(cell);
			sol_counter = 0;

			fe_values_momentum.reinit(cell_momentum);
			fe_values_def_grad.reinit(cell_def_grad);
			fe_values_momentum.get_function_values(sol_n_plus_1_momentum, sol_vec_momentum);
			fe_values_def_grad.get_function_values(sol_n_plus_1_def_grad, sol_vec_def_grad);


			for (const unsigned int q_point : fe_values_pressure.quadrature_point_indices())
			{
				for (unsigned int i = 0; i < dim; i++) { //Extracts momentum values, puts them in vector form

					temp_momentum[i] = sol_vec_momentum[q_point](i);
					//temp_momentum_residual[i] = residual_vec[q_point](sol_counter);
				}
				
				Jf = get_Jf(FF);
				HH = unit_symmetric_tensor<dim>();

				//cout << "HH : " << HH << std::endl;
				//cout << " FF : " << FF << std::endl;
				//temp_momentum += tau_pJ * temp_momentum_residual;

				//cout << "Jf : " << Jf << std::endl;
				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					cell_rhs(i) += -scalar_product(fe_values_pressure[Pressure].gradient(i, q_point), transpose(HH) * temp_momentum) *
						fe_values_pressure.JxW(q_point);
					//cout << "cell_rhs values : " << cell_rhs(i) << std::endl;
				}
			}
			//for (const auto& face : cell->face_iterators())
			//{

			//	if (face->at_boundary())
			//	{

			//		fe_face_values_pressure.reinit(cell, face);
			//		fe_face_values_momentum.reinit(cell_momentum, face);
			//		fe_face_values_def_grad.reinit(cell_def_grad, face);
			//		fe_face_values_momentum.get_function_values(sol_n_plus_1_momentum, face_sol_vec_momentum);
			//		fe_face_values_def_grad.get_function_values(sol_n_plus_1_def_grad, face_sol_vec_def_grad);

			//		if (face->boundary_id() == 5) {
			//			for (const unsigned int q_point : fe_face_values_pressure.quadrature_point_indices())
			//			{
			//				face_temp_momentum = 0;
			//				face_FF = 0;
			//				Jf = 0;
			//				face_HH = 0;

			//				for (int i = 0; i < dim; i++) {
			//					face_temp_momentum[i] = face_sol_vec_momentum[q_point](i);							}
			//				sol_counter = 0;
			//				for (int i = 0; i < dim; i++) {
			//					for (int j = 0; j < dim; j++) {
			//						face_FF[i][j] = face_sol_vec_def_grad[q_point](sol_counter);
			//						++sol_counter;
			//					}
			//				}

			//				Jf = get_Jf(face_FF);
			//				face_HH = get_HH(face_FF, Jf);
			//				//cofactor = reinterpret_cast<pointhistory<dim>*>(cell->user_pointer())[q_point].cofactor_store;
			//				for (unsigned int i = 0; i < dofs_per_cell; ++i)
			//				{
			//					cell_rhs(i) += fe_face_values_pressure[Pressure].value(i, q_point) *
			//						transpose(face_HH) *
			//						face_temp_momentum *
			//						Jf *
			//						fe_face_values_pressure.normal_vector(q_point) *
			//						fe_face_values_pressure.JxW(q_point);
			//				}

			//			}
			//		}
			//	}
			//}

			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				pressure_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
			cell_momentum++;
			cell_def_grad++;
		}
	}

	template <int dim>
	void Incompressible<dim>::assemble_momentum_rhs(Vector<double>& sol_n_pressure, Vector<double>& sol_n_plus_1_pressure)
	{
		momentum_rhs = 0;

		FEValuesExtractors::Vector Momentum(0);

		FEValues<dim> fe_values_momentum(mapping_simplex,
			fe_momentum,
			quadrature_formula_momentum,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEValues<dim> fe_values_pressure(mapping_simplex,
			fe_pressure,
			quadrature_formula_pressure,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		const unsigned int dofs_per_cell = fe_momentum.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula_momentum.size();

		std::vector<double> sol_vec_pressure(n_q_points);
		std::vector<double> old_sol_vec_pressure(n_q_points);





		Vector<double>     cell_rhs(dofs_per_cell);


		//Defines vectors to contain values for physical parameters


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		double temp_pressure;
		double old_temp_pressure;
		Tensor<2, dim> HH;
		double Jf;
		//double old_pressure;

		auto cell_pressure = dof_handler_pressure.begin_active();
		for (const auto& cell : dof_handler_momentum.active_cell_iterators())
		{


			HH = 0;
			temp_pressure = 0;
			old_temp_pressure = 0;
			//old_pressure = 0;
			cell_rhs = 0;
			fe_values_momentum.reinit(cell);
			fe_values_pressure.reinit(cell_pressure);

			fe_values_pressure.get_function_values(sol_n_plus_1_pressure, sol_vec_pressure);

			for (const unsigned int q_point : fe_values_momentum.quadrature_point_indices())
			{
				HH = reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q_point].HH_store;
				temp_pressure = sol_vec_pressure[q_point];
				old_temp_pressure = old_sol_vec_pressure[q_point];

				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					cell_rhs(i) += -scalar_product(fe_values_momentum[Momentum].gradient(i, q_point), (temp_pressure - old_temp_pressure) * HH) *
						fe_values_momentum.JxW(q_point);
				}
			}
			

			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				momentum_rhs(local_dof_indices[i]) += cell_rhs(i);
			}
			cell_pressure++;
		}
	}

	template<int dim>
	void Incompressible<dim>::solve_ForwardEuler()
	{
		assemble_def_grad_rhs(momentum_old_solution);
		solve_F(def_grad_old_solution, def_grad_solution);
		assemble_momentum_int_rhs(def_grad_old_solution, pressure_old_solution);
		solve_momentum_int(momentum_old_solution, momentum_solution);
		assemble_pressure_rhs(momentum_solution, def_grad_old_solution);
		solve_p(pressure_old_solution, pressure_solution);
		cout << "Updating displacement" << std::endl;
		update_displacement(momentum_old_solution, 0.0, momentum_solution, 1.0);
		cout << std::endl;

	}


	template<int dim>
	void Incompressible<dim>::solve_ssprk2()
	{
	assemble_def_grad_rhs(momentum_old_solution);
	cout << "Norm of def grad rhs : " << def_grad_rhs.l2_norm() << std::endl;
	cout << "Solving for def grad" << std::endl;
	solve_F(def_grad_old_solution, def_grad_int_solution);
	cout << "Assembling intermediate momentum rhs" << std::endl;
	assemble_momentum_int_rhs(def_grad_old_solution, pressure_old_solution);
	cout << "Norm of intermediate momentum rhs : " << momentum_rhs.l2_norm() << std::endl;
	cout << "Solving for intermediate momentum" << std::endl;
	solve_momentum_int(momentum_old_solution, momentum_int_solution);
	cout << "Assembling pressure rhs" << std::endl;
	assemble_pressure_rhs(momentum_int_solution, def_grad_old_solution);
	cout << "Norm of pressure rhs : " << pressure_rhs.l2_norm() << std::endl;
	cout << "Solving for pressure" << std::endl;
	solve_p(pressure_old_solution, pressure_int_solution);
	cout << "Updating displacement" << std::endl;
	update_displacement(momentum_old_solution, 0.0, momentum_int_solution, 1.0);
	cout << std::endl;

	assemble_pressure_Lap(def_grad_int_solution);
	assemble_def_grad_rhs(momentum_int_solution);
	cout << "Norm of def grad rhs : " << def_grad_rhs.l2_norm() << std::endl;
	cout << "Solving for def grad" << std::endl;
	solve_F(def_grad_int_solution, def_grad_solution);
	cout << "Assembling intermediate momentum rhs" << std::endl;
	assemble_momentum_int_rhs(def_grad_int_solution, pressure_int_solution);
	cout << "Norm of intermediate momentum rhs : " << momentum_rhs.l2_norm() << std::endl;
	cout << "Solving for intermediate momentum" << std::endl;
	solve_momentum_int(momentum_int_solution, momentum_solution);
	cout << "Assembling pressure rhs" << std::endl;
	assemble_pressure_rhs(momentum_solution, def_grad_int_solution);
	cout << "Norm of pressure rhs : " << pressure_rhs.l2_norm() << std::endl;
	cout << "Solving for pressure" << std::endl;
	solve_p(pressure_int_solution, pressure_solution);
	cout << "Updating displacement" << std::endl;

	momentum_solution = 0.5 * momentum_old_solution + 0.5 * momentum_solution;
	def_grad_solution = 0.5 * def_grad_old_solution + 0.5 * def_grad_solution;
	pressure_solution = 0.5 * pressure_old_solution + 0.5 * pressure_solution;

	update_displacement(momentum_old_solution, 0.5, momentum_solution, 0.5);
	cout << std::endl;
	}
//
//	//Assembles system, solves system, updates quad data.
//	template<int dim>
//	void Incompressible<dim>::solve_ssprk3()
//	{
//		cout << " Assembling system..." << std::flush;
//		assemble_system(old_solution);
//		assemble_def_grad_rhs(old_solution);
//		solve_F(old_solution, int_solution);
//		assemble_momentum_int_rhs(old_solution);
//		solve_momentum_int(old_solution, int_solution);
//		assemble_pressure_rhs(int_solution);
//		solve_p(old_solution, int_solution);
//		assemble_momentum_rhs(old_solution, int_solution);
//		solve_momentum(old_solution, int_solution);
//		update_displacement(old_solution, 0.0, solution, 1.0);
//		cout << std::endl;
//
//
//		assemble_system(int_solution);
//		assemble_def_grad_rhs(int_solution);
//		solve_F(int_solution, int_solution_2);
//		assemble_momentum_int_rhs(int_solution);
//		solve_momentum_int(int_solution, int_solution_2);
//		assemble_pressure_rhs(int_solution_2);
//		solve_p(int_solution, int_solution_2);
//		assemble_momentum_rhs(int_solution, int_solution_2);
//		solve_momentum(int_solution, int_solution_2);
//		int_solution_2 = 0.75 * old_solution + 0.25 * int_solution_2;
//		update_displacement(old_solution, 0.75, solution, 0.25);
//		cout << std::endl;
//
//
//		cout << std::endl;
//		assemble_system(int_solution_2);
//		assemble_def_grad_rhs(int_solution_2);
//		solve_F(int_solution_2, solution);
//		assemble_momentum_int_rhs(int_solution_2);
//		 solve_momentum_int(int_solution_2, solution);
//		assemble_pressure_rhs(solution);
//		solve_p(int_solution_2, solution);
//		assemble_momentum_rhs(int_solution_2, solution);
//		solve_momentum(int_solution_2, solution);
//		solution = 1.0 / 3.0 * old_solution + 2.0 / 3.0 * solution;
//		update_displacement(old_solution, 1.0 / 3.0, solution, 2.0 / 3.0);
//		cout << std::endl;
//
//
//	}


	//solves system using direct solver
	template <int dim>
    void Incompressible<dim>::solve_F(Vector<double>& def_grad_sol_n, Vector<double>& def_grad_sol_n_plus_1)
	{


		//residual.block(2) = 0;

		std::vector<bool> Def_Grad(dim * dim); //Should be Tensor<2>, but component_mask only works with symmetrics or vectors
		for (unsigned int i = 0; i < dim * dim; ++i) {
			Def_Grad[i] = true;
		}

		SparseMatrix<double>& un_M = unconstrained_mass_matrix_def_grad;
		const auto op_un_M = linear_operator(un_M);
		Vector<double> un_rhs = def_grad_rhs;

		Vector<double> old_sol = def_grad_sol_n;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, old_sol);

		AffineConstraints<double> all_constraints;





		AffineConstraints<double> F_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
			dof_handler_def_grad,
			1,
			Functions::ZeroFunction<dim>(dim*dim),
			F_constraints,
			fe_def_grad.component_mask(Def_Grad));
		F_constraints.close();
		all_constraints.merge(F_constraints);
		all_constraints.close();
		auto setup_constrained_rhs = constrained_right_hand_side(
			all_constraints, op_un_M, un_rhs);

		Vector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M2 = constrained_mass_matrix_def_grad;


		auto& def_grad = def_grad_sol_n_plus_1;



		Vector<double> F_rhs = rhs;


		SolverControl            solver_control(1000, 1e-16 * def_grad_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> F_preconditioner;
		F_preconditioner.initialize(M2, 1.2);

		//cout << "norm of right hand side : " << def_grad_rhs.l2_norm() << std::endl;

		solver.solve(M2, def_grad, F_rhs, F_preconditioner);
		all_constraints.distribute(def_grad_sol_n_plus_1);

		// For updating the residual

//		Vector<double> F_DQ = (sol_n_plus_1.block(2) - sol_n.block(2));
//		F_DQ /= present_timestep;
		//residual.block(2) = F_rhs - F_DQ;

		//return solver_control.last_step();
	}
	template <int dim>
	void Incompressible<dim>::solve_momentum_int(Vector<double>& momentum_sol_n, Vector<double>& momentum_sol_n_plus_1)
	{

        SparseMatrix<double>& un_M = unconstrained_mass_matrix_momentum;
		const auto op_un_M = linear_operator(un_M);
        Vector<double> un_rhs = momentum_rhs;
		Vector<double> old_sol = momentum_sol_n;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, old_sol);

		AffineConstraints<double> all_constraints;

        FEValuesExtractors::Vector Momentum(0);


		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex,
                                                         dof_handler_momentum,
                                                         1,
                                                         Functions::ZeroFunction<dim>(dim),
                                                         u_constraints,
                                                         fe_momentum.component_mask(Momentum));
		u_constraints.close();

		all_constraints.merge(u_constraints);
		all_constraints.close();
		auto setup_constrained_rhs = constrained_right_hand_side(
			all_constraints, op_un_M, un_rhs);

		Vector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M0 = constrained_mass_matrix_momentum;


		auto& momentum = momentum_sol_n_plus_1;



		Vector<double> u_rhs = rhs;

		/*SparseDirectUMFPACK M0_direct;
		M0_direct.initialize(M0);
		M0_direct.vmult(momentum, u_rhs);*/
		SolverControl            solver_control(1000, 1e-16 * momentum_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		//cout << "norm of right hand side : " << momentum_rhs.l2_norm() << std::endl;

		PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
		u_preconditioner.initialize(M0, 1.2);

		solver.solve(M0, momentum, u_rhs, u_preconditioner);
		all_constraints.distribute(momentum_sol_n_plus_1);

		// For updating the residual
//		Vector<double> u_DQ = (sol_n_plus_1.block(0) - sol_n.block(0));
//		u_DQ /= present_timestep;
		//residual.block(0) = u_rhs - u_DQ;


	}

	//solves system using direct solver
	template <int dim>
	void Incompressible<dim>::solve_p(Vector<double>& pressure_sol_n, Vector<double>& pressure_sol_n_plus_1)
	{

		//residual.block(1) = 0;

		SparseMatrix<double>& un_M = unconstrained_it_matrix_pressure;
		const auto op_un_M = linear_operator(un_M);
		Vector<double> un_rhs = pressure_rhs;

		Vector<double> old_sol = pressure_sol_n;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, old_sol);




		AffineConstraints<double> p_constraints;
		//dealii::VectorTools::interpolate_boundary_values(mapping_simplex, dof_handler,
		//	4,
		//	Functions::ZeroFunction<dim>(dim + 1 + dim * dim),
		//	p_constraints,
		//	fe.component_mask(Pressure));
		p_constraints.close();

		auto setup_constrained_rhs = constrained_right_hand_side(
			p_constraints, op_un_M, un_rhs);
		Vector<double> rhs;
		rhs.reinit(old_sol);
		setup_constrained_rhs.apply(rhs);


		const auto& M1 = constrained_it_matrix_pressure;


		auto& pressure = pressure_sol_n_plus_1;



		Vector<double> p_rhs = rhs;

		/*SparseDirectUMFPACK M1_direct;
		M1_direct.initialize(M1);
		M1_direct.vmult(pressure, p_rhs);*/

		//cout << "norm of right hand side : " << pressure_rhs.l2_norm() << std::endl;
		SolverControl            solver_control(100000, 1e-10 * pressure_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> p_preconditioner;
		p_preconditioner.initialize(M1, 1.2);

		solver.solve(M1, pressure, p_rhs, p_preconditioner);

		p_constraints.distribute(pressure_sol_n_plus_1);

		//For the residuals
//		Vector<double> p_DQ = (sol_n_plus_1.block(1) - sol_n.block(1));
//		p_DQ /= present_timestep;
		//residual.block(1) = p_rhs - p_DQ;

//		return solver_control.last_step();

	}


	//solves system using direct solver
	template <int dim>
	void Incompressible<dim>::solve_momentum(Vector<double>& sol_n_momentum, Vector<double>& sol_n_plus_1_momentum)
	{

		FEValuesExtractors::Vector Momentum(0);

		SparseMatrix<double>& un_M = unconstrained_mass_matrix_momentum;
		const auto op_un_M = linear_operator(un_M);
		Vector<double> un_rhs = momentum_rhs;
		auto& sol = sol_n_plus_1_momentum;

		un_rhs *= present_timestep;
		un_M.vmult_add(un_rhs, sol);





		const auto& M0 = constrained_mass_matrix_momentum;


		auto& momentum = sol_n_plus_1_momentum;

    

		AffineConstraints<double> u_constraints;
		dealii::VectorTools::interpolate_boundary_values(mapping_simplex, dof_handler_momentum,
			1,
			Functions::ZeroFunction<dim>(dim),
			u_constraints,
			fe_momentum.component_mask(Momentum));
		u_constraints.close();

		auto setup_constrained_rhs = constrained_right_hand_side(
			u_constraints, op_un_M, un_rhs);
		Vector<double> rhs;
		rhs.reinit(sol);
		setup_constrained_rhs.apply(rhs);

		Vector<double> u_rhs = rhs;

		/*SparseDirectUMFPACK M0_direct;
		M0_direct.initialize(M0);
		M0_direct.vmult(momentum, u_rhs);*/

		//cout << "norm of right hand side : " << momentum_rhs.l2_norm() << std::endl;

		SolverControl            solver_control(1000, 1e-16 * momentum_rhs.l2_norm());
		SolverCG<Vector<double>>  solver(solver_control);

		PreconditionJacobi<SparseMatrix<double>> u_preconditioner;
		u_preconditioner.initialize(M0, 1.2);

		solver.solve(M0, momentum, u_rhs, u_preconditioner);
		u_constraints.distribute(sol_n_plus_1_momentum);

//		return solver_control.last_step();
	}









	//Spits out solution into vectors then into .vtks
	template<int dim> 
	void Incompressible<dim>::output_results(Vector<double>& momentum_solution,
		Vector<double>& pressure_solution,
		Vector<double>& def_grad_solution) const
	{
        
		const FESystem<dim> joint_fe(fe_momentum, 1, fe_momentum, 1, fe_pressure, 1, fe_def_grad, 1);
		DoFHandler<dim> joint_dof_handler(triangulation);
		joint_dof_handler.distribute_dofs(joint_fe);
		Vector<double> joint_solution(joint_dof_handler.n_dofs());
		std::vector <types::global_dof_index> local_joint_dof_indices(
			joint_fe.n_dofs_per_cell()),
			local_momentum_dof_indices(fe_momentum.n_dofs_per_cell()),
			local_pressure_dof_indices(fe_pressure.n_dofs_per_cell()),
			local_def_grad_dof_indices(fe_def_grad.n_dofs_per_cell());

		typename DoFHandler<dim>::active_cell_iterator
			joint_cell = joint_dof_handler.begin_active(),
			joint_end_cell = joint_dof_handler.end(),
			momentum_cell = dof_handler_momentum.begin_active(),
			pressure_cell = dof_handler_pressure.begin_active(),
			def_grad_cell = dof_handler_def_grad.begin_active();
		for (; joint_cell != joint_end_cell; ++joint_cell, ++momentum_cell, ++pressure_cell, ++def_grad_cell) {
			joint_cell->get_dof_indices(local_joint_dof_indices);
			momentum_cell->get_dof_indices(local_momentum_dof_indices);
			pressure_cell->get_dof_indices(local_pressure_dof_indices);
			def_grad_cell->get_dof_indices(local_def_grad_dof_indices);
			for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i) {
				switch (joint_fe.system_to_base_index(i).first.first)
				{
				case 0:
					Assert(joint_fe.system_to_base_index(i).second < local_momentum_dof_indices.size(), ExcInternalError());
					joint_solution(local_joint_dof_indices[i]) = momentum_solution(local_momentum_dof_indices[joint_fe.system_to_base_index(i).second]);
					break;
				case 1:
					Assert(joint_fe.system_to_base_index(i).second < local_momentum_dof_indices.size(), ExcInternalError());
					joint_solution(local_joint_dof_indices[i]) = total_displacement(local_momentum_dof_indices[joint_fe.system_to_base_index(i).second]);
					break;
				case 2: 
					Assert(joint_fe.system_to_base_index(i).second < local_pressure_dof_indices.size(), ExcInternalError());
					joint_solution(local_joint_dof_indices[i]) = pressure_solution(local_pressure_dof_indices[joint_fe.system_to_base_index(i).second]);
					break;
				case 3: 
					Assert(joint_fe.system_to_base_index(i).second < local_def_grad_dof_indices.size(), ExcInternalError());
					joint_solution(local_joint_dof_indices[i]) = def_grad_solution(local_def_grad_dof_indices[joint_fe.system_to_base_index(i).second]);
					break;
				default:
					Assert(false, ExcInternalError());
				}
			}
		}
		std::vector<std::string> joint_solution_names(dim, "momentum");
		std::vector<std::string> displacement_names(dim, "Displacement");
		joint_solution_names.insert(joint_solution_names.end(), displacement_names.begin(), displacement_names.end());
		joint_solution_names.emplace_back("pressure");
		std::vector<std::string> def_grad_names(dim * dim, "Deformation_Gradient");
		joint_solution_names.insert(joint_solution_names.end(), def_grad_names.begin(), def_grad_names.end());

		DataOut<dim> data_out;
		data_out.attach_dof_handler(joint_dof_handler);
		std::vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation(
			dim, DataComponentInterpretation::component_is_part_of_vector);
		std::vector<DataComponentInterpretation::DataComponentInterpretation> displacement_interpretation(
			dim, DataComponentInterpretation::component_is_part_of_vector);
		std::vector<DataComponentInterpretation::DataComponentInterpretation> pressure_interpretation(
			1, DataComponentInterpretation::component_is_scalar);
		std::vector<DataComponentInterpretation::DataComponentInterpretation> def_grad_interpretation(
			dim * dim, DataComponentInterpretation::component_is_part_of_tensor);
		component_interpretation.insert(component_interpretation.end(), displacement_interpretation.begin(), displacement_interpretation.end());
		component_interpretation.insert(component_interpretation.end(), pressure_interpretation.begin(), pressure_interpretation.end());
		component_interpretation.insert(component_interpretation.end(), def_grad_interpretation.begin(), def_grad_interpretation.end());

		data_out.add_data_vector(joint_solution,
			joint_solution_names,
			DataOut<dim>::type_dof_data,
			component_interpretation);
		data_out.build_patches(1);
		std::ofstream output("output-" + std::to_string(timestep_no) + ".vtu");
		data_out.write_vtu(output);
        

	}
	





	template <int dim>
	void Incompressible<dim>::do_timestep()
	{
		present_time += present_timestep;
		++timestep_no;
		cout << "_____________________________________________________________" << std::endl;
		cout << "Timestep " << timestep_no << " at time " << present_time
			<< std::endl;
		if (present_time > end_time)
		{
			present_timestep -= (present_time - end_time);
			present_time = end_time;
		}

		if (parameters.rk_order == 1)
		{
			solve_ForwardEuler();
		}
		else if (parameters.rk_order == 2)
		{
			solve_ssprk2();
		}
//		else if (parameters.rk_order == 3)
//		{
//			solve_ssprk3();
//		}
		if (abs(present_time - save_counter * save_time) < 0.1 * present_timestep) {
			cout << "Saving results at time : " << present_time << std::endl;
			output_results(momentum_solution, pressure_solution, def_grad_solution);
			save_counter++;
		}
		std::swap(momentum_old_solution, momentum_solution);
		std::swap(pressure_old_solution, pressure_solution);
		std::swap(def_grad_old_solution, def_grad_solution);

		cout << std::endl << std::endl;
	}


	template<int dim>
	void Incompressible<dim>::update_displacement(const Vector<double>& sol_n_momentum, const double& coeff_n, const Vector<double>& sol_n_plus_1_momentum, const double& coeff_n_plus)
	{
		auto old_momentum = sol_n_momentum;
		auto momentum = sol_n_plus_1_momentum;

		if (coeff_n != 0.0) {
			total_displacement -= incremental_displacement;
		}
		incremental_displacement = present_timestep * (coeff_n * old_momentum + coeff_n_plus * momentum);
		total_displacement += incremental_displacement;

	}



	

	// This chunk of code allows for communication between current code state and quad point history
	template<int dim>
	void Incompressible<dim>::setup_quadrature_point_history()
	{
		triangulation.clear_user_data();
		std::vector<PointHistory<dim>> tmp;
		quadrature_point_history.swap(tmp);
		quadrature_point_history.resize(triangulation.n_active_cells() * quadrature_formula_momentum.size());
		unsigned int history_index = 0;
		for (auto& cell : triangulation.active_cell_iterators())
			if (cell->is_locally_owned())
			{
				cell->set_user_pointer(&quadrature_point_history[history_index]);
				history_index += quadrature_formula_momentum.size();
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

