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
#include <deal.II/fe/fe.h>
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

//For timer
#include <deal.II/base/timer.h>


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
			double rho_inf;
			double alpha;
			int integrator;
			int n_ref;
			unsigned int velocity_order;
			unsigned int pressure_order;
			int max_it;
			bool LumpMass;
			bool Simplex;
			double e_tol;
			double c_tau;
			int max_ref;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Numerical::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				prm.declare_entry("rho_inf",
					"1.0",
					Patterns::Double(),
					"rho_inf");
				prm.declare_entry("alpha",
					"0.0",
					Patterns::Double(),
					"alpha");
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
				prm.declare_entry("max_it",
					"100",
					Patterns::Integer(0),
					"max_it");
				prm.declare_entry("LumpMass",
					"false",
					Patterns::Bool(),
					"LumpMass");
				prm.declare_entry("Simplex",
					"false",
					Patterns::Bool(),
					"Simplex");
				prm.declare_entry("e_tol",
					"0.000001",
					Patterns::Double(),
					"e_tol");
				prm.declare_entry("c_tau",
					"0.001",
					Patterns::Double(),
					"c_tau");
				prm.declare_entry("max_ref",
					"6",
					Patterns::Integer(0),
					"max_ref");
			}
			prm.leave_subsection();
		}
		void Numerical::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				rho_inf = prm.get_double("rho_inf");
				alpha = prm.get_double("alpha");
				integrator = prm.get_integer("Time integrator");
				n_ref = prm.get_integer("n_ref");
				velocity_order = prm.get_integer("Velocity order");
				pressure_order = prm.get_integer("Pressure order");
				max_it = prm.get_double("max_it");
				LumpMass = prm.get_bool("LumpMass");
				Simplex = prm.get_bool("Simplex");
				e_tol = prm.get_double("e_tol");
				c_tau = prm.get_double("c_tau");
				max_ref = prm.get_double("max_ref");
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
		full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3. * full_HH / Jf) + (pressure * full_HH);

		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

		//stress = mu * FF + (pressure * HH);
		//stress = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 2.0 * HH / Jf) + (pressure *HH);


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
		InitialVelocity<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		Assert(values.size() == (dim + 1), ExcDimensionMismatch(values.size(), dim + 1));
		values[0] = -velocity * std::sin(M_PI * p[dim - 1] / 12.) * p[1];
		values[1] = velocity * std::sin(M_PI * p[dim - 1] / 12.) * p[0];
		if (dim == 3) {
			values[2] = 0;
		}
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
		values[0] = a * time * time * time / 6.0;
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
		values[0] = 0;
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


	template <int dim>
	class Incompressible
	{
	public:
		Incompressible(const std::string& input_file);
		~Incompressible();
		void run();

	private:
		void         create_simplex_grid(Triangulation<2>& triangulation);
		void         create_simplex_grid(Triangulation<3>& triangulation);
		void         create_grid();
		void		 set_simulation_parameters();
		void		 set_stability_parameters();
		void		 set_integration_parameters();
		void         setup_system();
		void         assemble_system_IM();
		void         assemble_system_Kuu();
		void         assemble_system_not_Kuu();
		void		 assemble_Rp();
		void		 assemble_R();
		void         solve_semi_implicit();
		void		 solve_implicit();
		void		 solve_explicit();
		void		 solve_SI_system();
		void		 solve_EX_system();
		void		 solve_IM_system();
		void		 update_motion();
		void         output_results() const;
		void		 calculate_error();
		void		 do_timestep();



		Parameters::AllParameters parameters;
		int integrator;

		Triangulation<dim> triangulation;
		DoFHandler<dim>    dof_handler;

		std::unique_ptr<FiniteElement<dim>> v_base_fe;
		std::unique_ptr<FiniteElement<dim>> p_base_fe;
		std::unique_ptr<FESystem<dim>> fe_ptr;
		std::unique_ptr<MappingFE<dim>> mapping_ptr;




		AffineConstraints<double> constraints;
		AffineConstraints<double> displacement_constraints;
		AffineConstraints<double> pressure_constraints;


		std::unique_ptr<Quadrature<dim>> quad_rule_ptr;
		std::unique_ptr<Quadrature<dim - 1>> face_quad_rule_ptr;



		BlockSparsityPattern sparsity_pattern;
		BlockSparsityPattern un_sparsity_pattern;
		BlockSparseMatrix<double> K;
		BlockSparseMatrix<double> un_K;


		BlockVector<double> R;
		BlockVector<double> un_R;


		BlockVector<double> solution;
		BlockVector<double> solution_increment;
		BlockVector<double> old_solution;
		BlockVector<double> residual;




		Vector<double> displacement;
		Vector<double> old_displacement;
		Vector<double> velocity;
		Vector<double> old_velocity;
		Vector<double> acceleration;
		Vector<double> old_acceleration;
		Vector<double> pressure;
		Vector<double> old_pressure;

		Vector<double> u_dot;
		Vector<double> old_u_dot;
		Vector<double> a_nplusalpha;
		Vector<double> v_nplusbeta;


		BlockVector<double> true_solution;
		BlockVector<double> error;

		BlockVector<double> extra_vector;
		BlockVector<double> extra_vector2;

		double present_time;
		double dt;
		double rho_0;
		double end_time;
		double save_time;
		double save_counter;
		unsigned int timestep_no;
		unsigned int savestep_no;
		double pressure_mean;


		Vector<double> u_cell_wise_error;
		Vector<double> p_cell_wise_error;

		Vector<double> displacement_error_output;
		Vector<double> velocity_error_output;
		Vector<double> pressure_error_output;

		double E;
		double nu;
		unsigned int height;

		double rho_inf;
		double alpha;
		double beta;
		double gamma;


		double kappa;
		double mu;
		double c_kappa;
		double c_mu;

		double dt_tau;
		double tau;
		double h_min;

		//TimerOutput timer;


	};
	// Constructor for the main class
	template<int dim>
	Incompressible<dim>::Incompressible(const std::string& input_file)
		: parameters(input_file)
		, integrator(parameters.integrator)
		, dof_handler(triangulation)
		, timestep_no(0)
		, savestep_no(0)
		//, timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
	{
		if (parameters.Simplex == false) {
			quad_rule_ptr = std::make_unique<QGauss<dim>>(3);
			face_quad_rule_ptr = std::make_unique<QGauss<dim - 1>>(3);
			v_base_fe = std::make_unique<FE_Q<dim>>(parameters.velocity_order);
			p_base_fe = std::make_unique<FE_Q<dim>>(parameters.pressure_order);

		}
		else {
			quad_rule_ptr = std::make_unique<QGaussSimplex<dim>>(3 + 1);
			face_quad_rule_ptr = std::make_unique<QGaussSimplex<dim - 1>>(3);
			if (parameters.LumpMass == true) // Makes simplex space have bubbles so mass lumping can occur
			{
				v_base_fe = std::make_unique<FE_SimplexP_Bubbles<dim>>(parameters.velocity_order);
				p_base_fe = std::make_unique<FE_SimplexP<dim>>(parameters.pressure_order);
			}
			else
			{
				v_base_fe = std::make_unique<FE_SimplexP<dim>>(parameters.velocity_order);
				p_base_fe = std::make_unique<FE_SimplexP<dim>>(parameters.pressure_order);
			}
		}


		fe_ptr = std::make_unique<FESystem<dim>>(*v_base_fe, dim, *p_base_fe, 1);
		mapping_ptr = std::make_unique<MappingFE<dim>>(*v_base_fe);
	}


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
		int max_it = parameters.max_ref;
		std::vector<double> l1_u_eps_vec(max_it);
		std::vector<double> l2_u_eps_vec(max_it);
		std::vector<double> linfty_u_eps_vec(max_it);
		std::vector<double> l2_p_eps_vec(max_it);
		std::vector<double> l1_p_eps_vec(max_it);
		std::vector<double> linfty_p_eps_vec(max_it);
		set_simulation_parameters();
		set_integration_parameters();

		for (unsigned int ref_step = 0; ref_step < max_it; ++ref_step) {
			if (ref_step == 0) {
				if (parameters.Simplex == true) {
					create_simplex_grid(triangulation);
				}
				else {
					create_grid();
				}
			}
			set_stability_parameters();
			for (unsigned int i = 0; i < ref_step; ++i) {
				dt *= 0.5;
			}
			setup_system();
			extra_vector.block(0) = velocity;
			extra_vector2.block(0) = acceleration;
			output_results();

			save_counter = 1;
			if (integrator != 2) {
				{
					//TimerOutput::Scope timer_section(timer, "Assemble Kuu & Kpp");
					assemble_system_Kuu();
					cout << "Mass matrix assembled" << std::endl;

				}
			}
			
			cout << "New time step size : " << dt << std::endl;
			cout << std::endl;

			while (present_time < end_time - 1e-12) {
				do_timestep();
			}
			l2_u_eps_vec[ref_step] = displacement_error_output[0];
			l1_u_eps_vec[ref_step] = displacement_error_output[1];
			linfty_u_eps_vec[ref_step] = displacement_error_output[2];
			l2_p_eps_vec[ref_step] = pressure_error_output[0];
			l1_p_eps_vec[ref_step] = pressure_error_output[1];
			linfty_p_eps_vec[ref_step] = pressure_error_output[2];

			cout << "Chopping time step in half after iteration " << ref_step << " : " << std::endl;
			cout << "Number of steps taken after this iteration : " << timestep_no << std::endl;
						
			cout << "New time step size : " << dt << std::endl;
			cout << std::endl;

			present_time = parameters.start_time;
			timestep_no = 0;
		}
		std::cout << "Error calculated to be : " << std::endl;
		cout << "|____dt____|_l2_u_diff_|_l1_u_diff_|linfu_diff|_l2p_diff_|_l1p_diff_|linfpdiff|" << std::endl;
		for (int i = 0; i < max_it; ++i)
			cout << "|" << parameters.dt << "*0.5^" << i << "|" << l2_u_eps_vec[i] - l2_u_eps_vec[i - 1] << "|" << l1_u_eps_vec[i] - l1_u_eps_vec[i - 1] << "|" << linfty_u_eps_vec[i] - linfty_u_eps_vec[i - 1] << "|" << l2_p_eps_vec[i] - l2_p_eps_vec[i - 1] << "|" << l1_p_eps_vec[i] - l1_p_eps_vec[i - 1] << "|" << linfty_p_eps_vec[i] - linfty_p_eps_vec[i - 1] << std::endl;

	}

	template <int dim>
	void Incompressible<dim>::set_simulation_parameters()
	{
		height = 6;
		cout << "height is :" << height << std::endl;
		E = parameters.E;
		nu = parameters.nu;
		mu = E / (2. * (1. + nu));
		kappa = E / (3. * (1 - 2. * nu));
		rho_0 = parameters.rho_0;
		present_time = parameters.start_time;
		dt = parameters.dt;
		end_time = parameters.end_time;
		save_time = parameters.save_time;
	}
	template <int dim>
	void Incompressible<dim>::set_stability_parameters()
	{
		c_mu = std::sqrt(mu / rho_0);
		c_kappa = std::sqrt((kappa + 4. / 3. * mu) / rho_0);
		double c_tau = parameters.c_tau;
		if (nu == 0.5) {
			dt_tau = h_min / c_mu;
		}
		else {
			dt_tau = h_min / c_mu;
		}
		// (Defensive) Only implement nonzero stabilization terms if both V and P FE spaces are linear
		tau = parameters.velocity_order == 1 ? 0.5 * c_tau * std::max(dt_tau / 100., std::min(dt_tau, dt)) : 0.;
		//tau = parameters.velocity_order == 1 ? c_tau * dt_tau / 100.: 0.;
		//cout << "c_mu is : " << c_mu << std::endl;
		//cout << "c_kappa is : " << c_kappa << std::endl;
		//cout << "Stabilization 'step' is : " << dt_tau << std::endl;
		//cout << "Stabilization parameter is : " << tau << std::endl;
	}

	template<int dim>
	void Incompressible<dim>::set_integration_parameters()
	{
		if (integrator != 2)
		{
			alpha = parameters.alpha;
			beta = alpha + 1. / 12.;
			gamma = alpha + 0.5;
		}
		else
		{
			rho_inf = parameters.rho_inf;
			alpha = 1. / (1. + rho_inf);
			beta = (3. - rho_inf) / (2. * (1. + rho_inf));
			gamma = 0.5 + alpha - beta;
		}
	}

	template <int dim>
	void Incompressible<dim>::create_simplex_grid(Triangulation<2>& triangulation)
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
	void Incompressible<dim>::create_simplex_grid(Triangulation<3>& triangulation)
	{
		Triangulation<dim> quad_triangulation;
		std::vector<Point<3>> vertices = {
			{-1.0 , -1.0 , 0.0} , {1.0, -1.0, 0.0}, {-1.0, 1.0 , 0.0} , {1.0, 1.0, 0.0}, {-1.0, -1.0, 2.0}, {1.0, -1.0, 2.0}, {-1.0, 1.0, 2.0}, {1.0,1.0, 2.0},
		};
		std::vector<std::array<int, GeometryInfo<3>::vertices_per_cell>>
			cell_vertices = { {{0, 1, 2, 3, 4, 5, 6, 7}} };
		for (int h = 1; h < height; ++h) {
			std::vector<Point<3>> new_vertices = { {-1.0, -1.0, 2. * (h + 1.)}, {1.0, -1.0, 2. * (h + 1.) }, {-1.0, 1.0, 2. * (h + 1.) }, {1.0, 1.0, 2. * (h + 1.)} };
			std::vector<std::array<int, GeometryInfo<3>::vertices_per_cell>>
				new_cell_vertices = { {{4 * h, 4. * h + 1, 4 * h + 2, 4 * h + 3, 4 * h + 4, 4 * h + 5, 4 * h + 6, 4 * h + 7}} };
			vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
			cell_vertices.insert(cell_vertices.end(), new_cell_vertices.begin(), new_cell_vertices.end());
		}


		const unsigned int n_cells = cell_vertices.size();

		std::vector<CellData<3>> cells(n_cells, CellData<3>());
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
					if (abs(face_center[2] - 0.0) < 0.00001) {
						face->set_boundary_id(1);
					}
				}
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);

		triangulation.refine_global(parameters.n_ref);
	}

	template <int dim>
	void Incompressible<dim>::create_grid()
	{

		GridGenerator::subdivided_hyper_rectangle(triangulation, { 1, 1,height }, { -1.,-1.,0. }, { 1.,1., 2. * double(height) });
		triangulation.refine_global(parameters.n_ref);
		for (const auto& cell : triangulation.active_cell_iterators())
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (abs(face_center[2] - 0.0) < 0.00001) {
						face->set_boundary_id(1);
					}
				}
	}






	template <int dim>
	void Incompressible<dim>::setup_system()
	{



		K.clear();
		dof_handler.distribute_dofs(*fe_ptr);

		std::vector<unsigned int> block_component(dim + 1, 0);
		block_component[dim] = 1;
		DoFRenumbering::component_wise(dof_handler, block_component);



		// HOMOGENEOUS CONSTRAINTS
		{
			constraints.clear();
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*mapping_ptr,
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocity));
			constraints.close();

		}


		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
		const types::global_dof_index n_u = dofs_per_block[0];
		const types::global_dof_index n_p = dofs_per_block[1];



		//std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
		//	<< "Total number of cells: " << triangulation.n_cells()
		//	<< std::endl
		//	<< "Number of degrees of freedom: " << dof_handler.n_dofs()
		//	<< " (" << n_u << '+' << n_p << ')' << std::endl;





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
		old_solution.reinit(dofs_per_block);
		residual.reinit(dofs_per_block);

		extra_vector.reinit(dofs_per_block);
		extra_vector2.reinit(dofs_per_block);



		pressure.reinit(n_p);
		old_pressure.reinit(n_p);


		a_nplusalpha.reinit(n_u);
		v_nplusbeta.reinit(n_u);

		true_solution.reinit(dofs_per_block);

		error.reinit(dofs_per_block);

		displacement.reinit(n_u);
		old_displacement.reinit(n_u);
		acceleration.reinit(n_u);
		old_acceleration.reinit(n_u);
		velocity.reinit(n_u);
		old_velocity.reinit(n_u);
		u_dot.reinit(n_u);
		old_u_dot.reinit(n_u);


		u_cell_wise_error.reinit(triangulation.n_active_cells());
		p_cell_wise_error.reinit(triangulation.n_active_cells());

		displacement_error_output.reinit(3);
		pressure_error_output.reinit(3);

		const FEValuesExtractors::Vector Velocity(0);

		VectorTools::interpolate(*mapping_ptr,
			dof_handler,
			InitialVelocity<dim>(parameters.InitialVelocity, mu),
			solution,
			(*fe_ptr).component_mask(Velocity));
		velocity = solution.block(0);

		VectorTools::interpolate(*mapping_ptr,
			dof_handler,
			InitialAcceleration<dim>(parameters.BodyForce, mu, dt),
			solution,
			(*fe_ptr).component_mask(Velocity));
		acceleration = solution.block(0);
		VectorTools::interpolate(*mapping_ptr, dof_handler, InitialSolution<dim>(parameters.BodyForce, mu), solution);
		pressure_mean = solution.block(1).mean_value();
	}

	template <int dim>
	void Incompressible<dim>::assemble_system_IM()
	{

		un_K = 0;
		un_R = 0;
		R = 0;
		K = 0;

		FEValues<dim> fe_values((*mapping_ptr),
			*fe_ptr,
			*quad_rule_ptr,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values((*mapping_ptr),
			*fe_ptr,
			*face_quad_rule_ptr,
			update_values |
			update_gradients |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = (*fe_ptr).n_dofs_per_cell();
		const unsigned int n_q_points = (*quad_rule_ptr).size();
		const unsigned int n_face_q_points = (*face_quad_rule_ptr).size();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);


		double rho_0 = parameters.rho_0;

		Vector<double>     cell_rhs(dofs_per_cell);

		FExt<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		TractionVector<dim> traction_vector;
		std::vector<Tensor<1, dim>> traction_values(n_face_q_points, Tensor<1, dim>());


		Tensor<2, dim> FF;
		Tensor<2, dim> FF_inv_T;
		Tensor<2, dim> HH;
		double Jf;
		Tensor<2, dim> pk1;

		Tensor<2, dim> old_FF;
		Tensor<2, dim> old_FF_inv_T;
		Tensor<2, dim> old_HH;
		double old_Jf;
		Tensor<2, dim> old_pk1;

		Tensor<2, dim> dpk1_dFF;

		double temp_pressure;
		double old_temp_pressure;
		double real_old_temp_pressure;
		Tensor<1, dim> temp_a;
		Tensor<1, dim> temp_v;

		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> face_displacement_grads(n_face_q_points, Tensor<2, dim>());

		std::vector<double> sol_vec_pressure((*quad_rule_ptr).size());
		std::vector<double> old_sol_vec_pressure((*quad_rule_ptr).size());
		std::vector<Tensor<1, dim>> sol_vec_a((*quad_rule_ptr).size());
		std::vector<Tensor<1, dim>> sol_vec_v((*quad_rule_ptr).size());
		std::vector<double> face_sol_vec_pressure(n_face_q_points);
		std::vector<Tensor<1, dim>> residual_u(n_q_points, Tensor<1, dim>());


		a_nplusalpha = alpha * acceleration + (1. - alpha) * old_acceleration;
		v_nplusbeta = beta * velocity + (1. - beta) * old_velocity;

		auto fake_sol = solution;
		auto fake_sol2 = solution;
		fake_sol.block(0) = a_nplusalpha;
		fake_sol2.block(0) = v_nplusbeta;
		//cout << v_nplusbeta.l2_norm() << std::endl;
		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			cell_rhs = 0;
			cell_mass_matrix = 0;
			fe_values.reinit(cell);

			fe_values[Velocity].get_function_gradients(old_solution, old_displacement_grads);
			fe_values[Pressure].get_function_values(old_solution, old_sol_vec_pressure);

			fe_values[Velocity].get_function_gradients(solution, displacement_grads);
			fe_values[Pressure].get_function_values(solution, sol_vec_pressure);
			fe_values[Velocity].get_function_values(fake_sol, sol_vec_a);
			fe_values[Velocity].get_function_values(fake_sol2, sol_vec_v);
			fe_values[Velocity].get_function_values(residual, residual_u);


			present_time -= dt - beta * dt;
			right_hand_side.rhs_vector_value_list(fe_values.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time, mu, kappa);
			present_time += dt - beta * dt;


			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				temp_a = sol_vec_a[q];
				temp_v = sol_vec_v[q];
				temp_pressure = sol_vec_pressure[q];
				FF = get_real_FF(displacement_grads[q]);
				FF_inv_T = invert(transpose(FF));
				Jf = get_Jf(FF);
				HH = Jf * FF_inv_T;
				pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);
				//temp_pressure -= pressure_mean;

				old_temp_pressure = old_sol_vec_pressure[q];
				real_old_temp_pressure = old_temp_pressure;
				old_FF = get_real_FF(old_displacement_grads[q]);
				old_FF_inv_T = invert(transpose(old_FF));
				old_Jf = get_Jf(old_FF);
				old_HH = old_Jf * old_FF_inv_T;
				old_pk1 = get_pk1(old_FF, mu, old_Jf, old_temp_pressure, old_HH);

				old_temp_pressure = beta * temp_pressure + (1. - beta) * old_temp_pressure;
				old_FF = beta * FF + (1. - beta) * old_FF;
				old_FF_inv_T = beta * FF_inv_T + (1. - beta) * old_FF_inv_T;
				old_Jf = beta * Jf + (1. - beta) * old_Jf;
				old_HH = beta * HH + (1. - beta) * old_HH;
				old_pk1 = beta * pk1 + (1. - beta) * old_pk1;

				for (const unsigned int j : fe_values.dof_indices())
				{
					auto Grad_u_j = fe_values[Velocity].gradient(j, q);
					dpk1_dFF = alpha * mu / std::cbrt(Jf * Jf) * (-(2. / 3. * scalar_product(FF_inv_T, Grad_u_j) * (FF - 1. / 3. * scalar_product(FF, FF) * FF_inv_T)) +
						Grad_u_j - 2. / 3. * scalar_product(FF, Grad_u_j) * FF_inv_T +
						1. / 3. * scalar_product(FF, FF) * FF_inv_T * Grad_u_j * FF_inv_T +
						temp_pressure * FF_inv_T * (Grad_u_j - transpose(Grad_u_j)) * FF_inv_T);
					for (const unsigned int i : fe_values.dof_indices())
					{

						auto Grad_u_i = fe_values[Velocity].gradient(i, q);
						Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
						double N_p_i = fe_values[Pressure].value(i, q);


						cell_mass_matrix(i, j) += (alpha * alpha / (beta * gamma * gamma * dt * dt) * rho_0 * N_u_i * fe_values[Velocity].value(j, q) + //Kuu
							scalar_product(Grad_u_i, dpk1_dFF) + //KFS 
							beta * scalar_product(Grad_u_i, (HH)*fe_values[Pressure].value(j, q)) + //Kup
							beta/(gamma * dt) *N_p_i *  scalar_product((HH), fe_values[Velocity].gradient(j, q)) + //Kpu
							beta / kappa * N_p_i * fe_values[Pressure].value(j, q)) //Kpp
							* fe_values.JxW(q);
						if (parameters.velocity_order == 1 && parameters.c_tau != 0) {
							cell_mass_matrix(i, j) += (tau * (HH * fe_values[Pressure].gradient(i, q)) * (HH * fe_values[Pressure].gradient(j, q)) -
								tau * (alpha * alpha / (beta * gamma * gamma * dt * dt) * rho_0 * (HH * fe_values[Pressure].gradient(i, q)) * fe_values[Velocity].value(j, q))) *
								fe_values.JxW(q);
						}
					}
					cell_rhs(j) += (-fe_values[Velocity].value(j, q) * temp_a +
						-scalar_product(fe_values[Velocity].gradient(j, q), old_pk1) +
						fe_values[Velocity].value(j, q) * rhs_values[q] +
						(HH * fe_values[Pressure].gradient(j, q)) * temp_v + (old_temp_pressure-real_old_temp_pressure)/kappa) * fe_values.JxW(q);
					if (parameters.velocity_order == 1 && parameters.c_tau != 0) {
						cell_rhs(j) += tau *((HH * fe_values[Pressure].gradient(j,q)) * residual_u[q]) *fe_values.JxW(q);
					}
				}
			}
			for (const auto& face : cell->face_iterators())
			{
				if (face->at_boundary())
				{
					fe_face_values.reinit(cell, face);

					fe_face_values[Velocity].get_function_values(fake_sol2, sol_vec_v);
					fe_face_values[Velocity].get_function_gradients(solution, displacement_grads);
					

					present_time -= dt - beta * dt;
					traction_vector.traction_vector_value_list(fe_face_values.get_quadrature_points(), traction_values, parameters.TractionMagnitude, present_time);
					present_time += dt - beta * dt;

					for (const unsigned int q : fe_face_values.quadrature_point_indices())
					{
						temp_v = sol_vec_v[q];
						FF = get_real_FF(displacement_grads[q]);
						FF_inv_T = invert(transpose(FF));
						Jf = get_Jf(FF);
						HH = Jf * FF_inv_T;
						for (const unsigned int i : fe_face_values.dof_indices())
						{
							cell_rhs(i) -= fe_face_values[Pressure].value(i, q) * (transpose(HH) * temp_v) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
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
	void Incompressible<dim>::assemble_system_Kuu()
	{

		K = 0;
		un_K = 0;
		R = 0;
		un_R = 0;

		FEValues<dim> fe_values(*mapping_ptr,
			*fe_ptr,
			*quad_rule_ptr,
			update_values |
			update_quadrature_points |
			update_JxW_values);
		const std::vector<Point<dim>> q_points = (*quad_rule_ptr).get_points();



		const unsigned int dofs_per_cell = (*fe_ptr).n_dofs_per_cell();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);


		double rho_0 = parameters.rho_0;
		bool lump_mass = parameters.LumpMass;

		double scale;
		if (parameters.integrator == 1)
			scale = alpha / (beta * dt * dt);
		else
			scale = 1.;
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
				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					for (unsigned int j = 0; j < dofs_per_cell; ++j)
					{
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
						Tensor<1, dim> N_p_i = fe_values[Velocity].value(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, i) += scale * rho_0 * N_p_i * fe_values[Velocity].value(j, q) * fe_values.JxW(q);

						}
					}
				}
				cell->get_dof_indices(local_dof_indices);
				constraints.distribute_local_to_global(cell_mass_matrix,
					local_dof_indices,
					K);
				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
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
						Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
						double N_p_i = fe_values[Pressure].value(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, j) += (scale * rho_0 * N_u_i * fe_values[Velocity].value(j, q) +
								-1. / kappa * N_p_i * fe_values[Pressure].value(j, q)) * fe_values.JxW(q);
						}
					}
				}
				cell->get_dof_indices(local_dof_indices);
				constraints.distribute_local_to_global(cell_mass_matrix,
					local_dof_indices,
					K);
				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					for (unsigned int j = 0; j < dofs_per_cell; ++j)
					{
						un_K.add(local_dof_indices[i], local_dof_indices[j], cell_mass_matrix(i, j));
					}
				}
			}
		}
	}


	template <int dim>
	void Incompressible<dim>::assemble_system_not_Kuu()
	{


		un_K.block(0, 1) = 0;
		un_K.block(1, 0) = 0;
		un_K.block(1, 1) = 0;
		un_R = 0;
		R = 0;
		K.block(0, 1) = 0;
		K.block(1, 0) = 0;
		K.block(1, 1) = 0;



		FEValues<dim> fe_values(*mapping_ptr,
			*fe_ptr,
			(*quad_rule_ptr),
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values(*mapping_ptr,
			*fe_ptr,
			(*face_quad_rule_ptr),
			update_values |
			update_gradients |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = (*fe_ptr).n_dofs_per_cell();
		const unsigned int n_q_points = (*quad_rule_ptr).size();
		const unsigned int n_face_q_points = (*face_quad_rule_ptr).size();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);



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
		double old_temp_pressure;
		Tensor<1, dim> temp_velocity;
		Tensor<1, dim> temp_acceleration;
		//Tensor<1, dim> temp_residual;

		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<double> sol_vec_pressure(n_q_points);
		std::vector<double> sol_old_pressure(n_q_points);
		std::vector<Tensor<1, dim>> residual_u(n_q_points, Tensor<1, dim>());
		std::vector<Tensor<1, dim>> sol_vec_velocity(n_q_points, Tensor<1, dim>());
		std::vector<Tensor<1, dim>> sol_vec_acceleration(n_q_points, Tensor<1, dim>());
		std::vector<double> residual_p(n_q_points);

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			cell_rhs = 0;
			cell_mass_matrix = 0;
			fe_values.reinit(cell);

			fe_values[Velocity].get_function_gradients(solution, displacement_grads);
			fe_values[Pressure].get_function_values(solution, sol_vec_pressure);
			fe_values[Pressure].get_function_values(old_solution, sol_old_pressure);
			fe_values[Velocity].get_function_values(residual, residual_u);
			fe_values[Pressure].get_function_values(residual, residual_p);
			fe_values[Velocity].get_function_values(extra_vector, sol_vec_velocity);
			fe_values[Velocity].get_function_values(extra_vector2, sol_vec_acceleration);

			present_time -= dt;  // Evaluate FExt at time t-1
			right_hand_side.rhs_vector_value_list(fe_values.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time, mu, kappa);
			present_time += dt;

			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				temp_pressure = sol_vec_pressure[q];
				old_temp_pressure = sol_old_pressure[q];
				temp_velocity = sol_vec_velocity[q];
				temp_acceleration = sol_vec_acceleration[q];

				FF = get_real_FF(displacement_grads[q]);
				Jf = get_Jf(FF);
				HH = get_HH(FF, Jf);
				pk1 = get_pk1(FF, mu, Jf, temp_pressure, HH);
				//temp_pressure -= pressure_mean;
				for (const unsigned int i : fe_values.dof_indices())
				{
					auto Grad_u_i = fe_values[Velocity].gradient(i, q);
					auto Grad_p_i = fe_values[Pressure].gradient(i, q);
					Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
					double N_p_i = fe_values[Pressure].value(i, q);
					for (const unsigned int j : fe_values.dof_indices())
					{
						cell_mass_matrix(i, j) += (scalar_product(Grad_u_i, transpose(HH) * fe_values[Pressure].value(j, q)) + //Kup
							2. / 3. * gamma / beta * N_p_i * scalar_product((HH), fe_values[Velocity].gradient(j, q)) -
							1. / kappa * N_p_i * fe_values[Pressure].value(j, q)) * fe_values.JxW(q); //Kpp;
						if (parameters.velocity_order == 1 && parameters.c_tau != 0.) {	//These terms come from linearizing the residuals
							cell_mass_matrix(i, j) -= dt * tau * (HH * fe_values[Pressure].gradient(i, q) * (HH * fe_values[Pressure].gradient(j, q))) * fe_values.JxW(q);  //Kpp stabilization term
						}
					}
					cell_rhs(i) += (-scalar_product(Grad_u_i, pk1) +
						N_u_i * rhs_values[q] +
						2. / 3. * (HH * Grad_p_i) * (dt * (1. - gamma / beta) * temp_velocity + dt * dt * (1. - 0.5 * gamma / beta) * temp_acceleration) -
						N_p_i * (1. / 3. * temp_pressure - 1. / 3. * old_temp_pressure)/kappa) * fe_values.JxW(q);
					if (parameters.velocity_order == 1 && parameters.c_tau != 0.) {
						cell_rhs(i) -= dt *tau * ((HH * fe_values[Pressure].gradient(i, q)) * residual_u[q]) * fe_values.JxW(q); //Ru stabilization term
					}
				}

			}


			for (const auto& face : cell->face_iterators())
			{
				if (face->at_boundary())
				{
					fe_face_values.reinit(cell, face);

					fe_face_values[Velocity].get_function_values(extra_vector, sol_vec_velocity);
					fe_face_values[Velocity].get_function_values(extra_vector2, sol_vec_acceleration);
					fe_face_values[Velocity].get_function_gradients(solution, displacement_grads);
					fe_face_values[Velocity].get_function_values(residual, residual_u);



					present_time -= dt;
					traction_vector.traction_vector_value_list(fe_face_values.get_quadrature_points(), traction_values, parameters.TractionMagnitude, present_time);
					present_time += dt;

					for (const unsigned int q : fe_face_values.quadrature_point_indices())
					{
						for (const unsigned int i : fe_face_values.dof_indices())
						{
							temp_velocity = sol_vec_velocity[q];
							temp_acceleration = sol_vec_acceleration[q];

							FF = get_real_FF(displacement_grads[q]);
							Jf = get_Jf(FF);
							HH = get_HH(FF, Jf);
							if (face->boundary_id() == 1) {
								cell_rhs(i) += fe_face_values[Velocity].value(i, q) * traction_values[q] * fe_face_values.JxW(q);

							}
							cell_rhs(i) -= 2. / 3. * fe_face_values[Pressure].value(i, q) * (transpose(HH) * (dt * (1 - gamma / beta) * temp_velocity + dt * dt * (1. - 0.5 * gamma / beta) * temp_acceleration)) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
								if (parameters.velocity_order == 1 && parameters.c_tau != 0.) {
									cell_rhs(i) += dt * tau * (fe_face_values[Pressure].value(i, q) * (transpose(HH) * residual_u[q]) * fe_face_values.normal_vector(q))* fe_face_values.JxW(q);
									cout << (transpose(HH) * residual_u[q]) << "  " << fe_face_values.normal_vector(q) << std::endl;
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
	void Incompressible<dim>::assemble_R()
	{

		un_R = 0;
		R = 0;



		FEValues<dim> fe_values(*mapping_ptr,
			*fe_ptr,
			(*quad_rule_ptr),
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values(*mapping_ptr,
			*fe_ptr,
			(*face_quad_rule_ptr),
			update_values |
			update_gradients |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = (*fe_ptr).n_dofs_per_cell();
		const unsigned int n_q_points = (*quad_rule_ptr).size();
		const unsigned int n_face_q_points = (*face_quad_rule_ptr).size();



		FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);



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
		std::vector<double> sol_vec_pressure((*quad_rule_ptr).size());
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
					cell_rhs(i) += (-scalar_product(fe_values[Velocity].gradient(i, q), pk1) +
						N_u_i * rhs_values[q]) * fe_values.JxW(q);

					//std::cout << cell_rhs(i) <<  std::endl;
				}
			}
			for (const auto& face : cell->face_iterators())
			{
				if (face->at_boundary())
				{
					fe_face_values.reinit(cell, face);



					present_time -= dt;
					traction_vector.traction_vector_value_list(fe_face_values.get_quadrature_points(), traction_values, parameters.TractionMagnitude, present_time);
					present_time += dt;

					for (const unsigned int q : fe_face_values.quadrature_point_indices())
					{
						for (const unsigned int i : fe_face_values.dof_indices())
						{
							if (face->boundary_id() == 1) {
								cell_rhs(i) += fe_face_values[Velocity].value(i, q) * traction_values[q] * fe_face_values.JxW(q);

							}
						}
					}
				}

			}
			//cout << cell_rhs<< std::endl;



			cell->get_dof_indices(local_dof_indices);
			constraints.distribute_local_to_global(
				cell_rhs,
				local_dof_indices,
				R);
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {

				un_R(local_dof_indices[i]) += cell_rhs(i);
			}

		}
	}


	template <int dim>
	void Incompressible<dim>::assemble_Rp()
	{


		un_R.block(1) = 0;
		R.block(1) = 0;

		FEValues<dim> fe_values(*mapping_ptr,
			*fe_ptr,
			(*quad_rule_ptr),
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

		FEFaceValues<dim> fe_face_values(*mapping_ptr,
			*fe_ptr,
			(*face_quad_rule_ptr),
			update_values |
			update_gradients |
			update_normal_vectors |
			update_quadrature_points |
			update_JxW_values);


		const unsigned int dofs_per_cell = (*fe_ptr).n_dofs_per_cell();
		const unsigned int n_q_points = (*quad_rule_ptr).size();
		const unsigned int n_face_q_points = (*face_quad_rule_ptr).size();



		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


		const FEValuesExtractors::Vector Velocity(0);
		const FEValuesExtractors::Scalar Pressure(dim);



		Vector<double>     cell_rhs(dofs_per_cell);

		FExt<dim> right_hand_side;
		std::vector<Tensor<1, dim>> rhs_values(n_q_points, Tensor<1, dim>());


		Tensor<2, dim> FF;
		Tensor<2, dim> HH;
		double Jf;
		Tensor<2, dim> pk1;


		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> face_displacement_grads(n_face_q_points, Tensor<2, dim>());
		std::vector<double> sol_vec_pressure((*quad_rule_ptr).size());
		std::vector<Tensor<1, dim>> sol_vec_velocity(n_q_points, Tensor<1, dim>());
		std::vector<double> face_sol_vec_pressure(n_face_q_points);
		std::vector<Tensor<1, dim>> residual_u(n_q_points, Tensor<1, dim>());





		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			cell_rhs = 0;
			fe_values.reinit(cell);

			fe_values[Velocity].get_function_gradients(solution, displacement_grads);
			fe_values[Pressure].get_function_values(solution, sol_vec_pressure);
			fe_values[Velocity].get_function_values(extra_vector, sol_vec_velocity);
			fe_values[Velocity].get_function_values(residual, residual_u);

			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				FF = get_real_FF(displacement_grads[q]);
				Jf = get_Jf(FF);
				HH = get_HH(FF, Jf);
				//temp_pressure -= pressure_mean;
				for (const unsigned int i : fe_values.dof_indices())
				{

					cell_rhs(i) += dt * (HH * fe_values[Pressure].gradient(i, q)) * sol_vec_velocity[q] * fe_values.JxW(q);
					if (parameters.velocity_order == 1 && parameters.c_tau != 0.) {
						cell_rhs(i) += dt * tau * ((HH * fe_values[Pressure].gradient(i, q)) * residual_u[q]) * fe_values.JxW(q); //Ru stabilization term
					}
					//std::cout << cell_rhs(i) <<  std::endl;
				}
			}
			for (const auto& face : cell->face_iterators())
			{
				if (face->at_boundary())
				{
					fe_face_values.reinit(cell, face);

					fe_face_values[Velocity].get_function_values(extra_vector, sol_vec_velocity);
					fe_face_values[Velocity].get_function_gradients(solution, displacement_grads);
					fe_face_values[Velocity].get_function_values(residual, residual_u);

					for (const unsigned int q : fe_face_values.quadrature_point_indices())
					{
						FF = get_real_FF(displacement_grads[q]);
						Jf = get_Jf(FF);
						HH = get_HH(FF, Jf);
						for (const unsigned int i : fe_face_values.dof_indices())
						{
							cell_rhs(i) -= dt * fe_face_values[Pressure].value(i, q) * (transpose(HH) * sol_vec_velocity[q]) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
							if (parameters.velocity_order == 1 && parameters.c_tau != 0.) {
								cell_rhs(i) -= dt *tau * fe_face_values[Pressure].value(i, q) * (transpose(HH) * residual_u[q]) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
							}
						}
					}
				}
			}


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
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*mapping_ptr,
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocity));
		}
		constraints.close();
		{
			//TimerOutput::Scope timer_section(timer, "Assembling RHS and blocks");
			assemble_system_not_Kuu();
		}
		{
			//TimerOutput::Scope timer_section(timer, "Solving system");
			solve_SI_system();
		}
		update_motion();
		calculate_error();

	}

	template<int dim>
	void Incompressible<dim>::solve_implicit()
	{
		{
			constraints.clear();
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*mapping_ptr,
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocity));
		}
		constraints.close();
		int it = 0;
		int max_it = parameters.max_it;
		bool residual_flag = false;
		while ((residual_flag == false && it < max_it)) {



			acceleration = (alpha / (beta * gamma * gamma * dt * dt)) * (displacement - old_displacement)
				- 1. / (beta * gamma * dt) * old_velocity
				+ (gamma - 1.) / (gamma)*old_acceleration
				+ (gamma - alpha) / (beta * gamma * gamma * dt) * old_u_dot;

			velocity = (alpha / (beta * gamma * dt)) * (displacement - old_displacement)
				+ (beta - 1.) / beta * old_velocity
				+ (gamma - alpha) / (beta * gamma) * old_u_dot;

			{
				//TimerOutput::Scope timer_section(timer, "Assembling system");
				assemble_system_IM();

			}


			/*a_nplusalpha *= -(dt * dt * gamma * gamma * beta) / (alpha * alpha);
			K.block(0, 0).vmult_add(R.block(0), a_nplusalpha);
			un_K.block(0, 0).vmult_add(un_R.block(0), a_nplusalpha);*/
			double e_tol = parameters.e_tol;
			if ((R.block(0).linfty_norm() > e_tol) || (R.block(1).linfty_norm() > e_tol)) {
				//cout << "residual norm of " << R.block(0).linfty_norm() << " and " << R.block(1).linfty_norm() << std::endl;
				{
					//TimerOutput::Scope timer_section(timer, "Solving");
					solve_IM_system();
				}

				solution.block(0) = displacement + solution_increment.block(0);
				solution.block(1) = pressure + solution_increment.block(1);
				displacement = solution.block(0);
				pressure = solution.block(1);


				/*cout << "a norm is : " << acceleration.linfty_norm() << std::endl;
				cout << "v norm is : " << velocity.linfty_norm() << std::endl;
				cout << "d norm is : " << solution.block(0).linfty_norm() << std::endl;
				cout << "delta d norm is : " << solution_increment.block(0).linfty_norm() << std::endl;
				cout << "udot norm is: " << u_dot.linfty_norm() << std::endl;
				cout << "p norm is : " << solution.block(1).linfty_norm() << std::endl;
				cout << std::endl;*/


				++it;
			}
			else {
				residual_flag = true;
			}

		}
		u_dot = 1. / (gamma * dt) * (displacement - old_displacement) + (gamma - 1.) / gamma * old_u_dot;

		//cout << "Newton solver converged in " << it << "/" << max_it << " iterations with residual norms of " << R.block(0).l2_norm() << " and " << R.block(1).l2_norm() << std::endl;
		update_motion();


	}

	template<int dim>
	void Incompressible<dim>::solve_explicit()
	{
		{
			constraints.clear();
			const FEValuesExtractors::Vector Velocity(0);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*mapping_ptr,
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocity));

		}
		constraints.close();
		{
			//TimerOutput::Scope timer_section(timer, "Assembling RHS");
			assemble_R();
		}
		{
			//TimerOutput::Scope timer_section(timer, "Solving system");
			solve_EX_system();
		}
		//calculate_error();

	}



	template <int dim>
	void Incompressible<dim>::solve_SI_system()
	{



		const auto& Kuu = K.block(0, 0);
		const auto& Kup = K.block(0, 1);
		const auto& Kpu = K.block(1, 0);
		const auto& Kpp = K.block(1, 1);


		const auto op_Kuu = linear_operator(Kuu);
		const auto op_Kup = linear_operator(Kup);
		const auto op_Kpu = linear_operator(Kpu);
		const auto op_Kpp = linear_operator(Kpp);

		auto& Ru = R.block(0);
		const auto& Rp = R.block(1);

		auto& delta_u = solution_increment.block(0);
		auto& delta_p = solution_increment.block(1);

		SolverControl reduction_control_Kuu(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kuu(reduction_control_Kuu);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kuu;
		preconditioner_Kuu.initialize(Kuu);


		SolverControl solver_control_S(30000, 1.0e-12);

		const auto op_Kuu_inv = inverse_operator(op_Kuu, solver_Kuu, preconditioner_Kuu);
		auto op_S = op_Kpp - op_Kpu * op_Kuu_inv * op_Kup;
		if (parameters.nu == 0.5) {
			op_S = op_Kpp - op_Kpu * op_Kuu_inv * op_Kup;
		}

		SolverGMRES<Vector<double>> solver_S(solver_control_S);

		PreconditionIdentity preconditioner_S;


		Vector<double> un_motion(acceleration.size());
		un_motion.add((1.0 - alpha / (2.0 * beta)), acceleration, -1.0 * alpha / (beta * dt), velocity);
		double scale = -beta * dt * dt / alpha;
		un_motion *= scale;
		Kuu.vmult_add(Ru, un_motion);

		//Solve for the pressure increment via shur complement
		const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);
		delta_p = op_S_inv * (Rp - op_Kpu * op_Kuu_inv * Ru);
		constraints.distribute(solution_increment);

		//Solve for velocity
		Ru -= op_Kup * delta_p;
		delta_u = op_Kuu_inv * (Ru);
		constraints.distribute(solution_increment);

		un_motion *= -1.;
		Kuu.vmult_add(Ru, un_motion);
	}

	template <int dim>
	void Incompressible<dim>::solve_EX_system()
	{



		const auto& Kuu = K.block(0, 0);
		const auto& Kpp = K.block(1, 1);
		const auto& Ru = R.block(0);
		const auto& Rp = R.block(1);

		SolverControl reduction_control_Kuu(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kuu(reduction_control_Kuu);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kuu;
		preconditioner_Kuu.initialize(Kuu, 1.2);

		SolverControl reduction_control_Kpp(20000, 1.0e-12);
		SolverCG<Vector<double>> solver_Kpp(reduction_control_Kpp);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kpp;
		preconditioner_Kpp.initialize(Kpp, 1.2);




		solver_Kuu.solve(Kuu, acceleration, Ru, preconditioner_Kuu);
		solution.block(0) = solution.block(0) + dt * old_velocity + dt * dt * ((0.5 - beta) * old_acceleration + beta * acceleration);
		velocity = old_velocity + dt * ((1 - gamma) * old_acceleration + gamma * acceleration);
		assemble_Rp();
		solver_Kpp.solve(Kpp, solution_increment.block(1), Rp, preconditioner_Kpp);
		//Solve for the pressure increment via shur complement
		solution.block(1) = solution.block(1) + solution_increment.block(1);


		//Solve for velocity

		constraints.distribute(solution);

		residual.block(0) = old_acceleration - R.block(0);
		residual.block(1) = R.block(1);

		old_velocity = velocity;
		old_acceleration = acceleration;


		extra_vector.block(0) = old_velocity;
		extra_vector2.block(0) = old_acceleration;

		

	}

	template <int dim>
	void Incompressible<dim>::solve_IM_system()
	{
		const auto& Kuu = K.block(0, 0);
		const auto& Kup = K.block(0, 1);
		const auto& Kpu = K.block(1, 0);
		const auto& Kpp = K.block(1, 1);

		/*cout << Kuu.l1_norm() << std::endl;
		cout << Kup.l1_norm() << std::endl;
		cout << Kpu.l1_norm() << std::endl;
		cout << Kpp.l1_norm() << std::endl;*/

		const auto op_Kuu = linear_operator(Kuu);
		const auto op_Kup = linear_operator(Kup);
		const auto op_Kpu = linear_operator(Kpu);
		const auto op_Kpp = linear_operator(Kpp);
		
		auto& Ru = R.block(0);
		const auto& Rp = R.block(1);

		/*cout << Ru.l1_norm() << std::endl;
		cout << Rp.l1_norm() << std::endl;*/

		auto& delta_u = solution_increment.block(0);
		auto& delta_p = solution_increment.block(1);

		SolverControl reduction_control_Kuu(100000, 1.0e-9); //These are not ideal parameters
		SolverGMRES<Vector<double>> solver_Kuu(reduction_control_Kuu);
		PreconditionJacobi<SparseMatrix<double>> preconditioner_Kuu;
		preconditioner_Kuu.initialize(Kuu);


		SolverControl solver_control_S(100000, 1.0e-9); //These are also not ideal parameters

		const auto op_Kuu_inv = inverse_operator(op_Kuu, solver_Kuu, preconditioner_Kuu);
		auto op_S = op_Kpp - op_Kpu * op_Kuu_inv * op_Kup;
		if (parameters.nu == 0.5) {
			op_S = op_Kpp - op_Kpu * op_Kuu_inv * op_Kup;
		}
		

		SolverGMRES<Vector<double>> solver_S(solver_control_S);

		IterationNumberControl iteration_number_control_aS(30, 1.e-18);
		SolverGMRES<Vector<double>> solver_aS(iteration_number_control_aS);
		PreconditionIdentity preconditioner_aS;
		const auto op_aS = op_Kpu * linear_operator(preconditioner_Kuu) * op_Kup;
		const auto preconditioner_S = inverse_operator(op_aS, solver_aS, preconditioner_aS);


		//Solve for the pressure increment via shur complement
		const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S); //CHANGE BACK ASAP
		delta_p = op_S_inv * (Rp - op_Kpu * op_Kuu_inv * Ru);
		constraints.distribute(solution_increment);

		//Solve for velocity
		Ru -= op_Kup * delta_p;
		delta_u = op_Kuu_inv * (Ru);
		constraints.distribute(solution_increment);

	}

	template<int dim>
	void Incompressible<dim>::update_motion()
	{
		if (parameters.integrator != 2) {
			swap(velocity, old_velocity);
			swap(acceleration, old_acceleration);

			//std::cout << solution.block(1) << std::endl;
			old_solution = solution;
			solution += solution_increment;
			//pressure_mean = solution.block(1).mean_value(); //Subtract off average of pressure
			//solution.block(1).add(-mean);
			acceleration = (solution_increment.block(0) - dt * old_velocity + dt * dt * (beta - 0.5) * old_acceleration);
			acceleration *= (1.0 / (dt * dt * beta));

			velocity = old_velocity + dt * ((1.0 - gamma) * old_acceleration + gamma * acceleration);
			residual.block(0) = old_acceleration - R.block(0);
			residual.block(1) = R.block(1);
		}
		else {
			old_solution = solution;
			old_velocity = velocity;
			old_acceleration = acceleration;
			old_displacement = displacement;
			old_u_dot = u_dot;
			residual.block(0) = R.block(0);
			residual.block(1) = R.block(1);
		}
		extra_vector.block(0) = old_velocity;
		extra_vector2.block(0) = old_acceleration;

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

		


		QTrapezoid<1>  q_trapez;
		QIterated<dim> quadrature(q_trapez, 5);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim + 1),
			u_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::L2_norm,
			&velocity_mask);

		displacement_error_output[0] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::L2_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim + 1),
			u_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::L1_norm,
			&velocity_mask);

		displacement_error_output[1] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::L1_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim + 1),
			u_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::Linfty_norm,
			&velocity_mask);

		displacement_error_output[2] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::Linfty_norm);


		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim+1),
			p_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::L2_norm,
			&pressure_mask);

		pressure_error_output[0] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::L2_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim + 1),
			p_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::L1_norm,
			&pressure_mask);

		pressure_error_output[1] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::L1_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			solution,
			Functions::ZeroFunction<dim>(dim + 1),
			p_cell_wise_error,
			(*quad_rule_ptr),
			VectorTools::Linfty_norm,
			&pressure_mask);

		pressure_error_output[2] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::Linfty_norm);

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
		std::ofstream output("output-" + std::to_string(savestep_no) + ".vtu");
		data_out.write_vtu(output);


	}






	template <int dim>
	void Incompressible<dim>::do_timestep()
	{
		++timestep_no;
		present_time += dt;

		/*cout << "_____________________________________________________________" << std::endl;
		cout << "Timestep " << timestep_no << " at time " << present_time
			<< std::endl;*/

		if (parameters.integrator == 0)
		{
			solve_explicit();
		}
		if (parameters.integrator == 1)
		{
			solve_semi_implicit();
		}
		if (parameters.integrator == 2)
		{
			solve_implicit();
		}


		if (present_time > end_time)
		{
			dt -= (present_time - end_time);
			present_time = end_time;
		}
		if (abs(present_time - save_counter * save_time) < 0.1 * dt) {
			cout << "Saving results at time : " << present_time << std::endl;

			++savestep_no;

			calculate_error();
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

		NonlinearElasticity::Incompressible<3> incompressible("parameter_file.prm");
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

