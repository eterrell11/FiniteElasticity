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
#include <deal.II/lac/sparse_ilu.h>
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
#include <deal.II/base/table_handler.h>





//Enables Petsc for distributing work across my whopping 4 cores
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>


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


//For parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
	using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


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
			int integrator;
			int n_ref;
			unsigned int velocity_order;
			unsigned int pressure_order;
			int max_NS_it;
			bool LumpMass;
			bool Simplex;
			double e_tol;
			unsigned int max_ref;
			static void declare_parameters(ParameterHandler& prm);
			void parse_parameters(ParameterHandler& prm);
		};
		void Numerical::declare_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
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
				prm.declare_entry("max_ref",
					"5",
					Patterns::Integer(0),
					"max_ref");
			}
			prm.leave_subsection();
		}
		void Numerical::parse_parameters(ParameterHandler& prm)
		{
			prm.enter_subsection("Numerical parameters");
			{
				integrator = prm.get_integer("Time integrator");
				n_ref = prm.get_integer("n_ref");
				velocity_order = prm.get_integer("Velocity order");
				pressure_order = prm.get_integer("Pressure order");
				max_NS_it = prm.get_double("max_it");
				LumpMass = prm.get_bool("LumpMass");
				Simplex = prm.get_bool("Simplex");
				e_tol = prm.get_double("e_tol");
				max_ref = prm.get_integer("max_ref");
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

	template <class MatrixType, class PreconditionerType>
	class InverseMatrix : public Subscriptor
	{
	public:
		InverseMatrix(const MatrixType& m,
			const PreconditionerType& preconditioner);

		void vmult(LA::MPI::Vector& dst, const LA::MPI::Vector& src) const;

	private:
		const SmartPointer<const MatrixType>         matrix;
		const SmartPointer<const PreconditionerType> preconditioner;
	};


	template <class MatrixType, class PreconditionerType>
	InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
		const MatrixType& m,
		const PreconditionerType& preconditioner)
		: matrix(&m)
		, preconditioner(&preconditioner)
	{}

	template <class MatrixType, class PreconditionerType>
	void InverseMatrix<MatrixType, PreconditionerType>::vmult(
		LA::MPI::Vector& dst,
		const LA::MPI::Vector& src) const
	{
		SolverControl            solver_control(src.size(), 1e-6 * src.l2_norm());
		SolverCG<LA::MPI::Vector> cg(solver_control);

		dst = 0;

		cg.solve(*matrix, dst, src, *preconditioner);
	}

	

	namespace ConstitutiveModels
	{
		template<int dim>
		class WVol : public Function<dim>
		{
		public:
			double	W_prime(const int& wvol_form, const double& Jf);
			double  W_prime_lin(const int& wvol_form, const double& Jf, const Tensor<2, dim>& HH, const Tensor<2, dim>& grad_v, const double& dt);
			Tensor<1, dim>  W_prime_lin_u(const int& wvol_form, const double& Jf, const Tensor<2, dim>& HH, const Tensor<1, dim>& v);
		};

		template<int dim>
		double	WVol<dim>::W_prime(const int& wvol_form, const double& Jf)
		{
			if (wvol_form == 0) {
				return (Jf - 1.);
			}
			else if (wvol_form == 1) {
				return std::log(Jf);
			}
		}
		template<int dim>
		double WVol<dim>::W_prime_lin(const int& wvol_form, const double& Jf, const Tensor<2, dim>& HH, const Tensor<2, dim>& grad_v, const double& dt)
		{
			if (wvol_form == 0) {
				return dt * scalar_product(HH, grad_v);
			}
			else if (wvol_form == 1) {
				return dt / Jf * scalar_product(HH, grad_v);
			}
		}
		template<int dim>
		Tensor<1, dim>  WVol<dim>::W_prime_lin_u(const int& wvol_form, const double& Jf, const Tensor<2, dim>& HH, const Tensor<1, dim>& v)
		{
			if (wvol_form == 0) {
				return transpose(HH) * v;
			}
			else if (wvol_form == 1) {
				return transpose(HH) * v / Jf;
			}
		}

	} //namespace ConstitutiveModels

	template <class PreconditionerType>
	class SchurComplement : public Subscriptor
	{
	public:
		SchurComplement(
			const LA::MPI::BlockSparseMatrix& system_matrix,
			const InverseMatrix<LA::MPI::SparseMatrix, PreconditionerType>& A_inverse,
			const LA::MPI::BlockVector& exemplar,
			const double& kappa);

		void vmult(LA::MPI::Vector& dst, const LA::MPI::Vector& src) const;

	private:
		const SmartPointer<const LA::MPI::BlockSparseMatrix> system_matrix;
		const SmartPointer<
			const InverseMatrix<LA::MPI::SparseMatrix, PreconditionerType>>
			A_inverse;
		const SmartPointer<const LA::MPI::BlockVector> exemplar;
		const double kappa;
		mutable LA::MPI::Vector tmp1, tmp2, tmp3;
	};

	template <class PreconditionerType>
	SchurComplement<PreconditionerType>::SchurComplement(
		const LA::MPI::BlockSparseMatrix& system_matrix, 
		const InverseMatrix<LA::MPI::SparseMatrix, PreconditionerType>& A_inverse, 
		const LA::MPI::BlockVector& exemplar,
		const double& kappa)
		: system_matrix(&system_matrix)
		, A_inverse(&A_inverse)
		, exemplar(&exemplar)
		, kappa(kappa)
		, tmp1(exemplar.block(0))
		, tmp2(exemplar.block(0))
		, tmp3(exemplar.block(1))
	{}


	template <class PreconditionerType>
	void
		SchurComplement<PreconditionerType>::vmult(LA::MPI::Vector& dst,
			const LA::MPI::Vector& src) const
	{
		system_matrix->block(0, 1).vmult(tmp1, src);
		A_inverse->vmult(tmp2, tmp1);
		system_matrix->block(1, 0).vmult(dst, -1.0 * tmp2);
		system_matrix->block(1, 1).vmult(tmp3, src);
		dst.add(1./kappa, tmp3);
	}

	//Function for defining Kappa
	template <int dim>
	double get_kappa(double& E, double& nu) {
		double tmp;
		tmp = E / (3. * (1. - 2. * nu));
		return tmp;
	}

	//Function for defining mu
	template <int dim>
	double get_mu(double& E, double& nu) {
		double tmp = E / (2. * (1. + nu));
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
		get_pk1_dev(Tensor<2, dim>& FF, const double& mu, double& Jf, Tensor<2, dim>& HH)
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
		//full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3. * full_HH / Jf);
		full_pk1_stress = mu * full_FF;
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
		//stress = mu * (std::cbrt(Jf) / Jf) * (FF - scalar_product(FF, FF) / 2.0 * HH / Jf) + kappa*((Jf-1) * HH);
		full_pk1_stress = mu * (std::cbrt(Jf) / Jf) * (full_FF - scalar_product(full_FF, full_FF) / 3.0 * full_HH / Jf) + kappa * ((Jf - 1) * full_HH);
		//full_pk1_stress =  mu*full_FF + kappa * ((Jf - 1) * full_HH);
		for (int i = 0; i < dim; ++i)
			for (int j = 0; j < dim; ++j)
				stress[i][j] = full_pk1_stress[i][j];

		//stress = mu * FF + kappa * ((Jf - 1) * HH);
		return stress;
	}



	

	

	// template<int dim>
	// class FExt : public Function<dim>
	// {
	// public:
	// 	virtual void rhs_vector_value(const Point<dim>& /*p*/, Tensor<1, dim>& values, double& a, double& present_time, double& /*mu*/, double& /*kappa*/)

	// 	{
	// 		//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
	// 		Assert(dim >= 2, ExcInternalError());

	// 		values[0] = a * present_time;
	// 		values[1] = 0;
	// 	}
	// 	virtual void
	// 		rhs_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& BodyForce, double& present_time, double& mu, double& kappa)
	// 	{
	// 		const unsigned int n_points = points.size();
	// 		Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
	// 		for (unsigned int p = 0; p < n_points; ++p)
	// 			FExt<dim>::rhs_vector_value(points[p], value_list[p], BodyForce, present_time, mu, kappa);
	// 	}
	// };

	template<int dim>
	class FExt : public Function<dim>
	{
	public:
		virtual void rhs_vector_value(const Point<dim>& p, Tensor<1, dim>& values, double& a, double& present_time, double& mu)

		{
			//Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
			Assert(dim >= 2, ExcInternalError());
			Point<dim> u;

			for (unsigned int j = 0; j < 20; j++) {
				u[0] = a * std::sin(M_PI * (u[0] + p[0])) * std::cos(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time);
				u[1] = -a * std::cos(M_PI * (u[0] + p[0])) * std::sin(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time);
			}

			//PK1 contributions: (make it 2mu-1 for the original, potentially wrong, version)
			values[0] = a * (2.0 * mu-1.) * M_PI * M_PI * std::sin(M_PI * (u[0] + p[0])) * std::cos(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time);
			values[1] = -a * (2.0 * mu-1.) * M_PI * M_PI * std::cos(M_PI * (u[0] + p[0])) * std::sin(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time);
			//Acceleration contributions:
			// values[0] += a * M_PI * M_PI * (-std::sin(M_PI * (u[0] + p[0])) * std::cos(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time)
			// 	+ a * 0.5 * M_PI * std::sin(M_PI * 2.0 * (u[0] + p[0])) * std::cos(M_PI * present_time) * std::cos(M_PI * present_time));
			// values[1] += a * M_PI * M_PI * (std::cos(M_PI * (u[0] + p[0])) * std::sin(M_PI * (u[1] + p[1])) * std::sin(M_PI * present_time)
			// 	+ a * 0.5 * M_PI * std::sin(M_PI * 2.0 * (u[1] + p[1])) * std::cos(M_PI * present_time) * std::cos(M_PI * present_time));
		}
		virtual void
			rhs_vector_value_list(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& value_list, double& BodyForce, double& present_time, double& mu)
		{
			const unsigned int n_points = points.size();
			Assert(value_list.size() == n_points, ExcDimensionMismatch(value_list.size(), n_points));
			for (unsigned int p = 0; p < n_points; ++p)
				FExt<dim>::rhs_vector_value(points[p], value_list[p], BodyForce, present_time, mu);
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
		values[0] = velocity * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]);
		values[1] = -velocity * M_PI * std::sin(M_PI * p[1]) * std::cos(M_PI * p[0]);
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
		Solution(double& present_time, double& velocity);
		virtual void vector_value(const Point<dim>& p,
			Vector<double>& values) const override;
		virtual void
			vector_value_list(const std::vector<Point<dim>>& points,
				std::vector<Vector<double>>& value_list) const override;
	private:
		const double time;
		const double a;
	};

	template <int dim>
	Solution<dim>::Solution(double& present_time, double& velocity)
		: Function<dim>(dim + 1),
		time(present_time),
		a(velocity)
	{}

	template <int dim>
	void
		Solution<dim>::vector_value(const Point<dim>& p,
			Vector<double>& values) const
	{
		//Assert(values.size() == (dim), ExcDimensionMismatch(values.size(), dim));
		Point<dim> u;

		for (unsigned int j = 0; j < 20; j++)
		{
			u[0] = a * std::sin(M_PI * (u[0] + p[0])) * std::cos(M_PI * (u[1] + p[1])) * std::sin(M_PI * time);
			u[1] = -a * std::cos(M_PI * (u[0] + p[0])) * std::sin(M_PI * (u[1] + p[1])) * std::sin(M_PI * time);
		}
		values[0] = a * std::sin(M_PI *( p[0]+u[0])) * std::cos(M_PI * (p[1]+u[1])) * std::sin(M_PI * time);
		values[1] = -a * std::sin(M_PI *( p[1] + u[1])) * std::cos(M_PI * (p[0]+u[0])) * std::sin(M_PI * time);
		if (dim == 3)
			values[2] = 0;
		values[dim] = 0;
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
		DirichletValues<dim>::vector_value(const Point<dim>& /*p*/,
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
		void		 create_simplex_grid(parallel::shared::Triangulation<2>& triangulation);
		void		 create_simplex_grid(parallel::shared::Triangulation<3>& triangulation);
		void		 create_grid();
		void		 set_simulation_parameters();
		void         setup_system();
		void         assemble_system_mass();
		void         assemble_system_SI();
		void		 assemble_Rp();
		void		 assemble_R();
		void         solve_SI();
		void		 solve_SI_system();
		void		 update_motion();
		void         output_results() const;
		void		 calculate_error();
		void		 create_error_table();
		void		 do_timestep();
		
		
		Parameters::AllParameters parameters;


		MPI_Comm mpi_communicator;
		int integrator;

		const unsigned int n_mpi_processes;
		const unsigned int this_mpi_process;
		ConditionalOStream pcout;



		parallel::shared::Triangulation<dim> triangulation;
		double cell_measure;
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


		std::vector<IndexSet> owned_partitioning;
		std::vector<IndexSet> relevant_partitioning;


		BlockSparsityPattern sparsity_pattern;
		LA::MPI::BlockSparseMatrix K;
		LA::MPI::BlockSparseMatrix P;


		LA::MPI::BlockVector R;


		LA::MPI::BlockVector solution;
		LA::MPI::BlockVector relevant_solution;
		LA::MPI::BlockVector error_solution_store;
		LA::MPI::BlockVector relevant_error_solution_store;
		LA::MPI::BlockVector solution_dot;
		LA::MPI::BlockVector relevant_solution_dot;
		LA::MPI::BlockVector old_solution;
		LA::MPI::BlockVector relevant_old_solution;




		
		LA::MPI::Vector velocity;
		LA::MPI::Vector old_velocity;


		LA::MPI::BlockVector true_solution;
		LA::MPI::BlockVector error;

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
		double velocity_error_output;
		Vector<double> pressure_error_output;

		double E;
		double nu;


		double kappa;
		double mu;

		unsigned int height;
		unsigned int n_ref;

		double tau;
		int max_it;
		std::vector<double> l1_u_eps_vec;
		std::vector<double> l2_u_eps_vec;
		std::vector<double> linfty_u_eps_vec;
		std::vector<double> l2_p_eps_vec;
		std::vector<double> l1_p_eps_vec;
		std::vector<double> linfty_p_eps_vec;


	};
	// Constructor for the main class
	template<int dim>
	Incompressible<dim>::Incompressible(const std::string& input_file)
		: parameters(input_file)
		, mpi_communicator(MPI_COMM_WORLD)
		, integrator(parameters.integrator)
		, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
		, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
		, pcout(std::cout, (this_mpi_process == 0))
		, triangulation(mpi_communicator)
		, dof_handler(triangulation)
		, timestep_no(0)
		, savestep_no(0)
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
		max_it = parameters.max_ref;
		l1_u_eps_vec.reserve(max_it);
		l2_u_eps_vec.reserve(max_it);
		linfty_u_eps_vec.reserve(max_it);
		l2_p_eps_vec.reserve(max_it);
		l1_p_eps_vec.reserve(max_it);
		linfty_p_eps_vec.reserve(max_it);
		height = 6;
		for ( int ref_step = 0; ref_step < max_it; ++ref_step) {
			set_simulation_parameters();
			for (int i = 0; i < ref_step; ++i) {
				dt *= 0.5;
				n_ref += 1;
			}
			if (parameters.Simplex == true) {
				triangulation.clear();
				create_simplex_grid(triangulation);
			}
			else {
				triangulation.clear();
				create_grid();
			}
		

			
			setup_system();
			savestep_no = 0;

			output_results();

			save_counter = 1;
			if (integrator != 2) {
				{
					//TimerOutput::Scope timer_section(timer, "Assemble Kuu & Kpp");
					assemble_system_mass();
					pcout << "Mass matrix assembled" << std::endl;

				}
			}

			pcout << "New time step size : " << dt << std::endl;
			pcout << std::endl;


			while (present_time < end_time - 1e-12) {
				do_timestep();
			}

			l2_u_eps_vec[ref_step] = displacement_error_output[0];
			l1_u_eps_vec[ref_step] = displacement_error_output[1];
			linfty_u_eps_vec[ref_step] = displacement_error_output[2];
			l2_p_eps_vec[ref_step] = pressure_error_output[0];
			l1_p_eps_vec[ref_step] = pressure_error_output[1];
			linfty_p_eps_vec[ref_step] = pressure_error_output[2];

			pcout << "Chopping time step in half after iteration " << ref_step << " : " << std::endl;
			pcout << "Number of steps taken after this iteration : " << timestep_no << std::endl;

			pcout << "New time step size : " << dt << std::endl;
			pcout << std::endl;

			present_time = parameters.start_time;
			timestep_no = 0;
			error_solution_store = solution;

		}
		create_error_table();
	}

	template <int dim>
	void Incompressible<dim>::set_simulation_parameters()
	{
		E = parameters.E;
		nu = parameters.nu;
		mu = get_mu<dim>(E, nu);
		kappa = get_kappa<dim>(E, nu);
		rho_0 = parameters.rho_0;
		present_time = parameters.start_time;
		dt = parameters.dt;
		n_ref = parameters.n_ref;
		end_time = parameters.end_time;
		save_time = parameters.save_time;
	}

	


	template <int dim>
	void Incompressible<dim>::create_simplex_grid(parallel::shared::Triangulation<2>& triangulation)
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
					if (face_center[0] == 1) {
						face->set_boundary_id(2);
					}
				}
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);
		triangulation.refine_global(n_ref);

	}

	template <int dim>
	void Incompressible<dim>::create_simplex_grid(parallel::shared::Triangulation<3>& triangulation)
	{
		cell_measure = 1;
		Triangulation<dim> quad_triangulation;
		std::vector<Point<3>> vertices = {
			{-1.0 , -1.0 , 0.0} , {1.0, -1.0, 0.0}, {-1.0, 1.0 , 0.0} , {1.0, 1.0, 0.0}, {-1.0, -1.0, 2.0}, {1.0, -1.0, 2.0}, {-1.0, 1.0, 2.0}, {1.0,1.0, 2.0},
		};
		std::vector<std::array<int, GeometryInfo<3>::vertices_per_cell>>
			cell_vertices = { {{0, 1, 2, 3, 4, 5, 6, 7}} };
		for (unsigned int h = 1; h < height; ++h) {
			std::vector<Point<3>> new_vertices = { {-1.0, -1.0, 2. * (double(h) + 1.)}, {1.0, -1.0, 2. *(double(h) + 1.) }, {-1.0, 1.0, 2. * (double(h) + 1.) }, {1.0, 1.0, 2. * (double(h) + 1.)} };
			std::vector<std::array<int, GeometryInfo<3>::vertices_per_cell>>
				new_cell_vertices = { {{int(4 * h),int( 4 * h + 1), int(4 * h + 2), int(4 * h + 3), int(4 * h + 4), int(4 * h + 5), int(4 * h + 6), int(4 * h + 7)}} };
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
		{
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (abs(face_center[2] - 0.0) < 0.00001) {
						face->set_boundary_id(1);
					}
				}
			cell_measure = std::min(cell_measure, cell->measure());
		}
		GridGenerator::convert_hypercube_to_simplex_mesh(quad_triangulation, triangulation);

		triangulation.refine_global(n_ref);
	}

	template <int dim>
	void Incompressible<dim>::create_grid()
	{
		cell_measure = 1;
		GridGenerator::hyper_cube(triangulation,0, 1.);
		triangulation.refine_global(n_ref);
		for (const auto& cell : triangulation.active_cell_iterators())
		{
			for (const auto& face : cell->face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					if (abs(face_center[1] - 0.0) < 0.00001) {
						face->set_boundary_id(0);
					}
					if (abs(face_center[0] - 0.0) < 0.00001) {
						face->set_boundary_id(1);
					}
					if (abs(face_center[1] - 1.0) < 0.00001) {
						face->set_boundary_id(2);
					}
					if (abs(face_center[0] - 1.0) < 0.00001) {
						face->set_boundary_id(3);
					}
				}
			cell_measure = std::min(cell_measure, cell->measure());
		}

	}






	template <int dim>
	void Incompressible<dim>::setup_system()
	{



		K.clear();
		P.clear();
		dof_handler.distribute_dofs(*fe_ptr);




		std::vector<unsigned int> block_component(dim + 1, 0);
		block_component[dim] = 1;
		DoFRenumbering::component_wise(dof_handler, block_component);


		


		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
		const types::global_dof_index n_u = dofs_per_block[0];
		const types::global_dof_index n_p = dofs_per_block[1];


		const FEValuesExtractors::Vector Velocity(0);
		// const FEValuesExtractors::Scalar Pressure(dim); 
		// BlockMask velocity_mask = (*fe_ptr).block_mask(Velocity);
		// BlockMask pressure_mask = (*fe_ptr).block_mask(Pressure);

		owned_partitioning.resize(2);
		owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
		owned_partitioning[1] = dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);
	
		const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
		relevant_partitioning.resize(2);
		relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
		relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u+n_p);

		

		// HOMOGENEOUS CONSTRAINTS
		constraints.clear();
			const FEValuesExtractors::Scalar Velocityx(0);
			const FEValuesExtractors::Scalar Velocityy(1);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityx));
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				3,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityx));	
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				0,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityy));
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				2,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityy));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	0,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	1,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	2,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	3,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			constraints.close();





		pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl
			<< "Total number of cells: " << triangulation.n_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< " (" << n_u << '+' << n_p << ')' << std::endl;





		//DYNAMIC SPARSITY PATTERNS

		BlockDynamicSparsityPattern dsp(relevant_partitioning);
		DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			constraints,
			false);
		SparsityTools::distribute_sparsity_pattern(
			dsp,
			Utilities::MPI::all_gather(mpi_communicator,
				dof_handler.locally_owned_dofs()),
			mpi_communicator,
			locally_relevant_dofs);
		sparsity_pattern.copy_from(dsp);


		solution.reinit(owned_partitioning,
			mpi_communicator);
		error_solution_store.reinit(owned_partitioning,
			mpi_communicator);
		solution_dot.reinit(owned_partitioning,
			mpi_communicator);
		old_solution.reinit(owned_partitioning,
			mpi_communicator);

		relevant_error_solution_store.reinit(owned_partitioning, relevant_partitioning,
			mpi_communicator);
		relevant_solution.reinit(owned_partitioning, relevant_partitioning,
			mpi_communicator);
		relevant_solution_dot.reinit(owned_partitioning,
			relevant_partitioning,
			mpi_communicator);
		relevant_old_solution.reinit(owned_partitioning,
			relevant_partitioning,
			mpi_communicator);


		R.reinit(owned_partitioning,
			mpi_communicator);

		K.reinit(
			owned_partitioning,
			dsp,
			mpi_communicator);


		P.reinit(
			owned_partitioning,
			dsp,
			mpi_communicator);




		true_solution.reinit(owned_partitioning, mpi_communicator);

		error.reinit(owned_partitioning, mpi_communicator);

		
		velocity.reinit(owned_partitioning[0], mpi_communicator);
		old_velocity.reinit(owned_partitioning[0], mpi_communicator);


		u_cell_wise_error.reinit(triangulation.n_active_cells());
		p_cell_wise_error.reinit(triangulation.n_active_cells());


		displacement_error_output.reinit(3);
		pressure_error_output.reinit(3);


		VectorTools::interpolate(*mapping_ptr,
			dof_handler,
			InitialVelocity<dim>(parameters.InitialVelocity, mu),
			solution_dot,
			(*fe_ptr).component_mask(Velocity));
		velocity = solution_dot.block(0);
		old_velocity = solution_dot.block(0);
		relevant_solution_dot = solution_dot;


		VectorTools::interpolate(*mapping_ptr, dof_handler, InitialSolution<dim>(parameters.BodyForce, mu), solution);
		relevant_solution = solution;
		relevant_old_solution = solution;
		pressure_mean = solution.block(1).mean_value();
	}


	template <int dim>
	void Incompressible<dim>::assemble_system_mass()
	{
		K = 0;
		R = 0;

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
		if (parameters.integrator == 1) {
			scale = rho_0 / dt;
		}
		else {
			scale = 1.;
		}
		if (lump_mass == true) {
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				if (cell->subdomain_id() == this_mpi_process)
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
								cell_mass_matrix(i, j) += N_p_i * fe_values[Pressure].value(j, q) * fe_values.JxW(q);

							}
						}
					}
					cell->get_dof_indices(local_dof_indices);
					constraints.distribute_local_to_global(cell_mass_matrix,
						local_dof_indices,
						K);
				}
			}
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				if (cell->subdomain_id() == this_mpi_process)
				{
					cell_mass_matrix = 0;
					fe_values.reinit(cell);


					for (const unsigned int q : fe_values.quadrature_point_indices())
					{


						for (const unsigned int i : fe_values.dof_indices())
						{
							Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
							for (const unsigned int j : fe_values.dof_indices())
							{
								cell_mass_matrix(i, i) += scale * N_u_i * fe_values[Velocity].value(j, q) * fe_values.JxW(q);

							}
						}
					}
					cell->get_dof_indices(local_dof_indices);
					constraints.distribute_local_to_global(cell_mass_matrix,
						local_dof_indices,
						K);
				}
			}
		}
		else {
			for (const auto& cell : dof_handler.active_cell_iterators())
			{
				if (cell->subdomain_id() == this_mpi_process)
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
								cell_mass_matrix(i, j) += (scale * N_u_i * fe_values[Velocity].value(j, q) +
									N_p_i * fe_values[Pressure].value(j, q)) * fe_values.JxW(q);
							}
						}
					}
					cell->get_dof_indices(local_dof_indices);
					constraints.distribute_local_to_global(cell_mass_matrix,
						local_dof_indices,
						K);
				}
			}
		}
		K.compress(VectorOperation::add);
	}


	template <int dim>
	void Incompressible<dim>::assemble_system_SI()
	{

		P = 0;
		R = 0;
		K.block(0, 1) = 0;
		K.block(1, 0) = 0;



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
		FullMatrix<double> cell_preconditioner_matrix(dofs_per_cell, dofs_per_cell);

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
		Tensor<2, dim> old_HH;
		Tensor<2, dim> HH_tilde;
		double Jf;
		Tensor<2, dim> pk1;
		Tensor<2, dim> pk1_dev;
		Tensor<2, dim> old_pk1_dev;
		Tensor<2, dim> pk1_dev_tilde;

		double scale;
		double shifter;
		if (present_time >  dt) {
			scale = 2. / 3.;
			shifter = 0 /*1./3.*/;
		}
		else {
			scale = 1.;
			shifter = 0.;
		}
		//double temp_pressure;
		Tensor<1,dim> un;
		Tensor<1,dim> old_un;

		std::vector<Tensor<2, dim>> displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> face_displacement_grads(n_face_q_points, Tensor<2, dim>());
		std::vector<Tensor<2, dim>> old_displacement_grads(n_q_points, Tensor<2, dim>());
		std::vector<double> sol_vec_pressure(n_q_points);
		std::vector<Tensor<1,dim>> sol_vec_displacement(n_q_points, Tensor<1,dim>());
		std::vector<Tensor<1,dim>> old_sol_vec_displacement(n_q_points, Tensor<1,dim>());
		std::vector<Tensor<1,dim>> face_sol_vec_displacement(n_face_q_points, Tensor<1, dim>());
		std::vector<Tensor<1,dim>> old_face_sol_vec_displacement(n_face_q_points, Tensor<1, dim>());

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			if (cell->subdomain_id() == this_mpi_process)
			{
				cell_rhs = 0;
				cell_mass_matrix = 0;
				cell_preconditioner_matrix = 0;
				fe_values.reinit(cell);

				solution.update_ghost_values();
				old_solution.update_ghost_values();

				fe_values[Velocity].get_function_gradients(relevant_solution, displacement_grads);
				fe_values[Velocity].get_function_gradients(relevant_old_solution, old_displacement_grads);
				fe_values[Pressure].get_function_values(relevant_solution, sol_vec_pressure);
				fe_values[Velocity].get_function_values(relevant_solution, sol_vec_displacement);
				fe_values[Velocity].get_function_values(relevant_old_solution, old_sol_vec_displacement);



				right_hand_side.rhs_vector_value_list(fe_values.get_quadrature_points(), rhs_values, parameters.BodyForce, present_time, mu);


				for (const unsigned int q : fe_values.quadrature_point_indices())
				{
					//temp_pressure = sol_vec_pressure[q];
					un = sol_vec_displacement[q];
					old_un = old_sol_vec_displacement[q];
					FF = get_real_FF(old_displacement_grads[q]);
					Jf = get_Jf(FF);
					old_HH = get_HH(FF, Jf);
					old_pk1_dev = get_pk1_dev(FF, mu, Jf, old_HH);

					FF = get_real_FF(displacement_grads[q]);
					Jf = get_Jf(FF);
					HH = get_HH(FF, Jf);
					pk1_dev = get_pk1_dev(FF, mu, Jf, HH);

					HH_tilde = 2. * HH - old_HH;
					pk1_dev_tilde = 2. * pk1_dev - old_pk1_dev;

					// if (present_time < dt*1.1)
					// {
					// 	LA::MPI::BlockVector solution_extrap;
					// 	solution_extrap.reinit(solution);
					// 	solution_extrap = solution + dt * solution_dot;
					// 	auto tmp_relevant_solution(relevant_solution);
						
					// 	tmp_relevant_solution = solution_extrap;
					// 	fe_values[Velocity].get_function_gradients(tmp_relevant_solution, displacement_grads);
					// 	FF = get_real_FF(displacement_grads[q]);
					// 	double tmp_Jf = get_Jf(FF);
					// 	HH_tilde = get_HH(FF,tmp_Jf);
					// 	pk1_dev_tilde = get_pk1_dev(FF, mu, tmp_Jf, HH_tilde);
					// }
					

					//temp_pressure -= pressure_mean;
					for (const unsigned int i : fe_values.dof_indices())
					{
						auto Grad_u_i = fe_values[Velocity].gradient(i, q);
						Tensor<1, dim> N_u_i = fe_values[Velocity].value(i, q);
						double N_p_i = fe_values[Pressure].value(i, q);
						auto Grad_p_i = fe_values[Pressure].gradient(i, q);
						for (const unsigned int j : fe_values.dof_indices())
						{
							cell_mass_matrix(i, j) += (scale * scalar_product(Grad_u_i, (HH_tilde)*fe_values[Pressure].value(j, q)) - //Kup
								(1. - shifter) * dt * N_p_i * scalar_product(HH, fe_values[Velocity].gradient(j, q))) * fe_values.JxW(q);
							cell_preconditioner_matrix(i,j) += (1./kappa * N_p_i * fe_values[Pressure].value(j,q) - (HH)*Grad_p_i * (HH * fe_values[Pressure].gradient(j,q) )) * fe_values.JxW(q);

						}
						cell_rhs(i) += (-scale * scalar_product(Grad_u_i, pk1_dev_tilde) +
							rho_0 * scale * N_u_i * rhs_values[q] +
							N_p_i * (Jf - 1.0) - shifter * Grad_p_i * transpose(HH) * (un - old_un)) * fe_values.JxW(q);
					}
				}
				for (const auto& face : cell->face_iterators())
				{
					if (face->at_boundary())
					{

						fe_face_values.reinit(cell, face);
						fe_face_values[Velocity].get_function_gradients(relevant_solution, face_displacement_grads);
						fe_face_values[Velocity].get_function_values(relevant_solution, face_sol_vec_displacement);
						fe_face_values[Velocity].get_function_values(relevant_old_solution, old_face_sol_vec_displacement);

						traction_vector.traction_vector_value_list(fe_face_values.get_quadrature_points(), traction_values, parameters.TractionMagnitude, present_time);

						for (const unsigned int q : fe_face_values.quadrature_point_indices())
						{
							un = face_sol_vec_displacement[q];
							old_un = old_face_sol_vec_displacement[q];
							FF = get_real_FF(face_displacement_grads[q]);
							Jf = get_Jf(FF);
							HH = get_HH(FF, Jf);
							for (const unsigned int i : fe_face_values.dof_indices())
							{
								cell_rhs(i) += shifter * fe_face_values[Pressure].value(i, q) * (transpose(HH) * (un - old_un)) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
								if (face->boundary_id() == 1) {
									cell_rhs(i) += scale * fe_face_values[Velocity].value(i, q) * traction_values[q] * fe_face_values.JxW(q);

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
				constraints.distribute_local_to_global(cell_preconditioner_matrix,
					local_dof_indices,
					P);
			}
		}
		K.compress(VectorOperation::add);
		R.compress(VectorOperation::add);
		P.compress(VectorOperation::add);
	}

	




	template<int dim>
	void Incompressible<dim>::solve_SI()
	{
		constraints.clear();
			const FEValuesExtractors::Scalar Velocityx(0);
			const FEValuesExtractors::Scalar Velocityy(1);
			const FEValuesExtractors::Scalar Pressure(dim);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				1,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityx));
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				3,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityx));	
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				0,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityy));
			VectorTools::interpolate_boundary_values(*(mapping_ptr),
				dof_handler,
				2,
				Functions::ZeroFunction<dim>(dim + 1),
				constraints,
				(*fe_ptr).component_mask(Velocityy));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	0,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	1,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	2,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			// VectorTools::interpolate_boundary_values(*(mapping_ptr),
			// 	dof_handler,
			// 	3,
			// 	Functions::ZeroFunction<dim>(dim + 1),
			// 	constraints,
			// 	(*fe_ptr).component_mask(Pressure));
			constraints.close();
		{
			assemble_system_SI();
		}
		{
			solve_SI_system();
		}


		update_motion();
		calculate_error();

	}




	template <int dim>
	void Incompressible<dim>::solve_SI_system()
	{

		//std::unique_ptr<PackagedOperation<Vector<double>>>  linear_operator_ptr;

		const auto& Kuu = K.block(0, 0);
		const auto& Kup = K.block(0, 1);
		const auto& Kpu = K.block(1, 0);
		auto& Kpp = K.block(1, 1);

		//Kpp /= kappa;

		auto& Ru = R.block(0);
		auto& Rp = R.block(1);

		auto& Pp = P.block(1,1);


		SolverControl reduction_control_Kuu(1000, 1.0e-12);
		SolverCG<LA::MPI::Vector> solver_Kuu(reduction_control_Kuu);
		PETScWrappers::PreconditionBlockJacobi preconditioner_Kuu;
		preconditioner_Kuu.initialize(Kuu);

		//LA::MPI::PreconditionAMG::AdditionalData data;
		PETScWrappers::PreconditionSOR preconditioner_S_comp;
		preconditioner_S_comp.initialize(Pp);

		PETScWrappers::PreconditionSOR preconditioner_S_in;
		preconditioner_S_in.initialize(Pp);

		const InverseMatrix<LA::MPI::SparseMatrix, PETScWrappers::PreconditionBlockJacobi>
			M_inverse(Kuu, preconditioner_Kuu);

		SolverControl solver_control_S(2000, 1.0e-13);
		SolverGMRES<LA::MPI::Vector> solver_S(solver_control_S);

		IterationNumberControl iteration_number_control_aS(30, 1.e-18);
		//SolverMinRes<Vector<double>> solver_aS(iteration_number_control_aS);
		PreconditionIdentity preconditioner_aS;
		

		SchurComplement<PETScWrappers::PreconditionBlockJacobi> schur_complement(
			K, M_inverse, R, kappa);

		LA::MPI::Vector un_motion(velocity);		
		un_motion = 0;
		LA::MPI::Vector tmp1(Ru);
		tmp1 = 0;
		//LA::MPI::Vector tmp2(Rp);

		if (present_time < 1.1*dt) {
			un_motion.add(1.0, velocity);
		}
		else
		{
			un_motion.add(4./3., velocity, -1./3., old_velocity);
		}
		K.block(0,0).vmult_add(Ru, un_motion);

		M_inverse.vmult(tmp1, Ru);
		tmp1 *= -1.0;
		Kpu.vmult_add(Rp, tmp1);

		auto& v = solution_dot.block(0);

		auto& p = solution.block(1);
		auto fake_solution = solution;
		constraints.set_zero(solution);
		constraints.set_zero(solution_dot);

		if (parameters.nu == 0.5)
		{
			solver_S.solve(schur_complement, p, R.block(1), preconditioner_S_in);
		}
		else
		{
			solver_S.solve(schur_complement, p, R.block(1), preconditioner_S_comp);
		}

		Kup.vmult(tmp1, p);
		Ru.add(-1.0, tmp1);
		solver_Kuu.solve(Kuu, v, Ru, preconditioner_Kuu);
		//Solve for velocity

		constraints.distribute(solution);
		constraints.distribute(solution_dot);

		relevant_solution = solution;
		relevant_solution_dot = solution_dot;

	}

	
	template<int dim>
	void Incompressible<dim>::update_motion()
	{
		old_velocity = velocity;
		velocity = solution_dot.block(0);

		auto solution_save = solution.block(0);

		if (present_time > dt) {
			//solution.block(0) = 1. / 3. * (2. * dt * velocity + 4. * solution_save - old_solution.block(0));
			solution.block(0) = 4./3.*solution_save;
			solution.block(0).add(2./3., velocity);
			solution.block(0).add(-1./3.,old_solution.block(0) );
		}
		else {
			solution.block(0).add(dt, solution_dot.block(0));
		}
		old_solution.block(0) = solution_save;
			
		pressure_mean = solution.block(1).mean_value(); //Subtract off average of pressure
		//solution.block(1).add(-mean);
		
		relevant_solution = solution;
		relevant_old_solution = old_solution;
	}






	template<int dim>
	void Incompressible<dim>::calculate_error()
	{
		//error_solution_store.update_ghost_values();
		//relevant_error_solution_store = solution - error_solution_store;
		//error_sol.update_ghost_values();
		//VectorTools::interpolate(dof_handler, Solution<dim>(present_time, parameters.TractionMagnitude, kappa), true_solution);
		//error = (true_solution - solution);
		const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
		const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);

		const FEValuesExtractors::Scalar Pressure(dim);
		const FEValuesExtractors::Vector Velocity(0);
		pressure_mean = solution.block(1).mean_value();
		//solution.block(1).add(-pressure_mean);

		QTrapezoid<1>  q_trapez;
		QIterated<dim> error_quadrature(q_trapez, 5);

		//u_cell_wise_error.update_ghost_values();
		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Solution<dim>(present_time, parameters.InitialVelocity),
			u_cell_wise_error,
			error_quadrature,
			VectorTools::L2_norm,
			&velocity_mask);

		displacement_error_output[0] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::L2_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Solution<dim>(present_time, parameters.InitialVelocity),
			u_cell_wise_error,
			error_quadrature,
			VectorTools::L1_norm,
			&velocity_mask);

		displacement_error_output[1] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::L1_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Solution<dim>(present_time, parameters.InitialVelocity),
			u_cell_wise_error,
			error_quadrature,
			VectorTools::Linfty_norm,
			&velocity_mask);

		displacement_error_output[2] = VectorTools::compute_global_error(triangulation,
			u_cell_wise_error,
			VectorTools::Linfty_norm);

		//present_time -= dt;
		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Functions::ZeroFunction<dim>(dim + 1),
			p_cell_wise_error,
			error_quadrature,
			VectorTools::L2_norm,
			&pressure_mask);

		pressure_error_output[0] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::L2_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Functions::ZeroFunction<dim>(dim + 1),
			p_cell_wise_error,
			error_quadrature,
			VectorTools::L1_norm,
			&pressure_mask);

		pressure_error_output[1] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::L1_norm);

		VectorTools::integrate_difference(*mapping_ptr,
			dof_handler,
			relevant_solution,
			Functions::ZeroFunction<dim>(dim + 1),
			p_cell_wise_error,
			error_quadrature,
			VectorTools::Linfty_norm,
			&pressure_mask);

		pressure_error_output[2] = VectorTools::compute_global_error(triangulation,
			p_cell_wise_error,
			VectorTools::Linfty_norm);
		//present_time += dt;

	}


	template<int dim>
	void Incompressible<dim>::create_error_table()
	{
		TableHandler error_table;
		dt = parameters.dt;
		for (int i = 0; i < max_it; ++i) {
			//cout << "|" << parameters.dt << "*0.5^" << i << "|" << l2_u_eps_vec[i] - l2_u_eps_vec[i - 1] << "|" << l1_u_eps_vec[i] - l1_u_eps_vec[i - 1] << "|" << linfty_u_eps_vec[i] - linfty_u_eps_vec[i - 1]
			//<< "|" << l2_p_eps_vec[i] - l2_p_eps_vec[i - 1] << "|" << l1_p_eps_vec[i] - l1_p_eps_vec[i - 1] << "|" << linfty_p_eps_vec[i] - linfty_p_eps_vec[i - 1] << std::endl;
			dt *= 0.5;
			error_table.add_value("dt ", dt);
			error_table.set_scientific("dt ", true);
			error_table.add_value("dEu_l2 ", l2_u_eps_vec[i]);
			error_table.set_scientific("dEu_l2 ", true);
			error_table.add_value("dEu_l1 ", l1_u_eps_vec[i]);
			error_table.set_scientific("dEu_l1 ", true);
			error_table.add_value("dEu_linf ", linfty_u_eps_vec[i]);
			error_table.set_scientific("dEu_linf ", true);
			error_table.add_value("dEp_l2 ", l2_p_eps_vec[i]);
			error_table.set_scientific("dEp_l2 ", true);
			error_table.add_value("dEp_l1 ", l1_p_eps_vec[i]);
			error_table.set_scientific("dEp_l1 ", true);
			error_table.add_value("dEp_linf ", linfty_p_eps_vec[i]);
			error_table.set_scientific("dEp_linf ", true);
		}
		std::string boi;
		std::string nu_str;
		if (parameters.BodyForce != 0)
			boi = "BF";
		if (parameters.TractionMagnitude != 0)
			boi = "TR";
		if (parameters.InitialVelocity != 0)
			boi = "IV";
		if (parameters.nu == 0.4)
			nu_str = "4";
		if (parameters.nu == 0.49)
			nu_str = "49";
		if (parameters.nu == 0.5)
			nu_str = "5";
		std::ofstream output("error_table" + boi + nu_str + ".csv");
		error_table.write_text(std::cout);
		//std::ofstream output("error_table.csv");
		std::ostringstream stream;
		stream << "dt" << ',' << "l2_u" << ',' << "l1_u" << ',' << "linf_u" << ',' << "l2_p" << ',' << "l1_p" << ',' << "linf_p" << '\n';
		dt = parameters.dt;
		for (int i = 1; i < max_it; ++i) {
			dt *= 0.5;
			stream << dt << ',' << abs(l2_u_eps_vec[i] - l2_u_eps_vec[i - 1]) << ',' << abs(l1_u_eps_vec[i] - l1_u_eps_vec[i - 1]) << ',' << abs(linfty_u_eps_vec[i] - linfty_u_eps_vec[i - 1])
				<< ',' << abs(l2_p_eps_vec[i] - l2_p_eps_vec[i - 1]) << ',' << abs(l1_p_eps_vec[i] - l1_p_eps_vec[i - 1]) << ',' << abs(linfty_p_eps_vec[i] - linfty_p_eps_vec[i - 1]) << '\n';
		}
		output << stream.str();
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
			relevant_solution,
			solution_names,
			interpretation);


		data_out.add_data_vector(u_cell_wise_error,
			"Displacement_error",
			DataOut<dim>::type_cell_data);
		data_out.add_data_vector(p_cell_wise_error,
			"Pressure_error",
			DataOut<dim>::type_cell_data);

		LA::MPI::BlockVector extra_vector = solution;
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

		Vector<float> subdomain(triangulation.n_active_cells());
		for (unsigned int i = 0; i < subdomain.size(); ++i)
			subdomain(i) = triangulation.locally_owned_subdomain();
		data_out.add_data_vector(subdomain, "subdomain");


		data_out.build_patches(1);
		std::string output("output-" + std::to_string(savestep_no) + ".vtu");
		data_out.write_vtu_in_parallel(output, mpi_communicator);


	}






	template <int dim>
	void Incompressible<dim>::do_timestep()
	{
		++timestep_no;
		present_time += dt;

		//cout << "_____________________________________________________________" << std::endl;
		//cout << "Timestep " << timestep_no << " at time " << present_time
		//	<< std::endl;

		if (parameters.integrator == 1)
		{
			solve_SI();
		}


		if (present_time > end_time)
		{
			dt -= (present_time - end_time);
			present_time = end_time;
		}
		if (abs(present_time - save_counter * save_time) < 0.1 * dt) {
			//cout << "Saving results at time : " << present_time << std::endl;
			++savestep_no;
			output_results();
			save_counter++;
		}
	}


}






//Establsihes namespace, calls PETSc, calls run function, and all is bueno
int main(int argc, char* argv[])
{
	try
	{
		using namespace dealii;
		using namespace NonlinearElasticity;

		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
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

