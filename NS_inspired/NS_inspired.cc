#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

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
				prm.declare_entry("Initial momentum",
					"0",
					Patterns::Double(),
					"Initial momentum");
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
				InitialVelocity = prm.get_double("Initial momentum");
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

    namespace EquationData
    {
        template <int dim>
        class MultiComponentFunction : public Function<dim>
        {
        public:
            MultiComponentFunction(const double initial_time = 0.);
            void set_component(const unsigned int d);

        protected:
            unsigned int comp;
        };

        template <int dim>
        MultiComponentFunction<dim>::MultiComponentFunction(
            const double initial_time)
            : Function<dim>(1, initial_time)
            , comp(0)
        {}


        template <int dim>
        void MultiComponentFunction<dim>::set_component(const unsigned int d)
        {
            Assert(d < dim, ExcIndexRange(d, 0, dim * dim)); //extended range for component to allow for dealing with FF
            comp = d;
        }


        //Boundary condition data for velocity
        template <int dim>
        class Velocity : public MultiComponentFunction<dim>
        {
        public:
            Velocity(const double initial_time = 0.0);

            virtual double value(const Point<dim>& p,
                const unsigned int component = 0) const override;

            virtual void value_list(const std::vector<Point<dim>>& points,
                std::vector<double>& values,
                const unsigned int component = 0) const override;
        };


        template <int dim>
        Velocity<dim>::Velocity(const double initial_time)
            : MultiComponentFunction<dim>(initial_time)
        {}


        template <int dim>
        void Velocity<dim>::value_list(const std::vector<Point<dim>>& points,
            std::vector<double>& values,
            const unsigned int) const
        {
            const unsigned int n_points = points.size();
            AssertDimension(values.size(), n_points);
            for (unsigned int i = 0; i < n_points; ++i)
                values[i] = Velocity<dim>::value(points[i]);
        }

        template <int dim>
        double Velocity<dim>::value(const Point<dim>& p, const unsigned int) const
        {
            /*if (this->comp == 0)
            {
                const double Um = 1.5;
                const double H = 4.1;
                return 4. * Um * p(1) * (H - p(1)) / (H * H);
            }
            else*/
                return 0.;  //Change if you don't want zero initial values for the velocity field
        }

        //Boundary condition data for velocity
        template <int dim>
        class DeformationGradient : public MultiComponentFunction<dim>
        {
        public:
            DeformationGradient(const double initial_time = 0.0);

            virtual double value(const Point<dim>& p,
                const unsigned int component = 0) const override;

            virtual void value_list(const std::vector<Point<dim>>& points,
                std::vector<double>& values,
                const unsigned int component = 0) const override;
        };


        template <int dim>
        DeformationGradient<dim>::DeformationGradient(const double initial_time)
            : MultiComponentFunction<dim>(initial_time)
        {}


        template <int dim>
        void DeformationGradient<dim>::value_list(const std::vector<Point<dim>>& points,
            std::vector<double>& values,
            const unsigned int) const
        {
            const unsigned int n_points = points.size();
            AssertDimension(values.size(), n_points);
            for (unsigned int i = 0; i < n_points; ++i)
                values[i] = DeformationGradient<dim>::value(points[i]);
        }

        template <int dim>
        double DeformationGradient<dim>::value(const Point<dim>& p, const unsigned int) const
        {
            if ((this->comp) % dim == 0)
            {
                return 1.
            }
            else
            return 0.;  //Change if you don't want zero initial values for the velocity field
        }

        
    } // namespace EquationData

    template<int dim> 
    class NavierStokesProjection
    {
    public:
        NavierStokesProjection(const std::string& input_file);
        void run();

    protected:

        const double       dt;
        const double       start_time;
        const double       end_time;

        double E;
        double nu;

        double kappa;
        double mu;

        EquationData::Velocity<dim>               vel_exact;
        std::map<types::global_dof_index, double> boundary_values;
        std::vector<types::boundary_id>           boundary_ids;

        Triangulation<dim> triangulation;

        //One FE for each element (uncoupled)
        FE_Q<dim> fe_momentum;
        FE_Q<dim> fe_def_grad;
        FE_Q<dim> fe_pressure;

        DoFHandler<dim> dof_handler_momentum;
        DofHandler<dim> dof_handler_def_grad
        DoFHandler<dim> dof_handler_pressure;

        QGauss<dim> quadrature_momentum;
        QGauss<dim> quadrature_def_grad;
        QGauss<dim> quadrature_pressure;

        SparsityPattern sparsity_pattern_momentum;
        SparsityPattern sparsity_pattern_pressure;
        SparsityPattern sparsity_pattern_def_grad;
        SparsityPattern sparsity_pattern_pres_vel;

        SparseMatrix<double> vel_it_matrix[dim]; //For the addition of lap+mass and variable advection
        SparseMatrix<double> vel_Mass;
        SparseMatrix<double> def_grad_Mass;
        SparseMatrix<double> pres_Laplace;
        SparseMatrix<double> pres_Mass;
        SparseMatrix<double> pres_iterative; //can now be used so only Laplace portion needs to be changed with updates to cofactor; no need to affect the mass matrix

        Vector<double> pres_n_minus_1;
        Vector<double> pres_n;
        Vector<double> pres_star;
        Vector<double> pres_RHS;

        Vector<double> u_n_minus_1[dim];
        Vector<double> u_n[dim];
        Vector<double> u_star[dim];
        Vector<double> u_RHS[dim];
        Vector<double> u_RHS_2[dim];

        Vector<double> FF_n_minus_1[dim * dim];
        Vector<double> FF_n[dim * dim];
        Vector<double> FF_star[dim * dim];
        Vector<double> FF_RHS[dim * dim];

        Vector<double> force[dim];
        Vector<double> u_tmp;
        Vector<double> pres_tmp;
        Vector<double> FF_temp;

        SparseILU<double>   prec_momentum[dim];
        SparseILU<double>   prec_pres_Laplace;
        SparseDirectUMFPACK prec_mass;
        SparseDirectUMFPACK prec_vel_mass;

        DeclException2(ExcInvalidTimeStep,
            double,
            double,
            << " The time step " << arg1 << " is out of range."
            << std::endl
            << " The permitted range is (0," << arg2 << ']');

        void create_triangulation_and_dofs(const unsigned int n_refines);

        void initialize();

        void interpolate_momentum();

        void diffusion_step(const bool reinit_prec);

        void projection_step(const bool reinit_prec);

        void update_pressure(const bool reinit_prec);

    private:
        unsigned int vel_max_its;
        unsigned int vel_Krylov_size;
        unsigned int vel_off_diagonals;
        unsigned int vel_update_prec;
        double       vel_eps;
        double       vel_diag_strength;

        void initialize_momentum_matrices();

        void initialize_pressure_matrices();

        void initialize_def_grad_matrices();

        using IteratorTuple =
            std::tuple<typename DoFHandler<dim>::active_cell_iterator,
            typename DoFHandler<dim>::active_cell_iterator>;

        using IteratorPair = SynchronousIterators<IteratorTuple>;

        void initialize_gradient_operator();

        struct InitGradPerTaskData
        {
            unsigned int                         d;
            unsigned int                         vel_dpc;
            unsigned int                         pres_dpc;
            unsigned int                         def_grad_dpc;
            FullMatrix<double>                   local_grad;
            std::vector<types::global_dof_index> vel_local_dof_indices;
            std::vector<types::global_dof_index> pres_local_dof_indices;

            InitGradPerTaskData(const unsigned int dd,
                const unsigned int vdpc,
                const unsigned int pdpc,
                const unsigned int FFdpc)
                : d(dd)
                , vel_dpc(vdpc)
                , pres_dpc(pdpc)
                , def_grad_dpc(FFdpc)
                , local_grad(vdpc, pdpc)
                , vel_local_dof_indices(vdpc)
                , pres_local_dof_indices(pdpc)
            {}
        };

        struct InitGradScratchData
        {
            unsigned int  nqp;
            FEValues<dim> fe_val_vel;
            FEValues<dim> fe_val_pres;
            FEValues<dim> fe_val_def_grad;
            InitGradScratchData(const FE_Q<dim>& fe_v,
                const FE_Q<dim>& fe_p,
                const FE_Q<dim>& fe_FF,
                const QGauss<dim>& quad,
                const UpdateFlags  flags_v,
                const UpdateFlags  flags_p,
                const UpdateFlags  flags_FF)
                : nqp(quad.size())
                , fe_val_vel(fe_v, quad, flags_v)
                , fe_val_pres(fe_p, quad, flags_p)
                , fe_val_def_grad(fe_FF, quad, flags_FF)
            {}
            InitGradScratchData(const InitGradScratchData& data)
                : nqp(data.nqp)
                , fe_val_vel(data.fe_val_vel.get_fe(),
                    data.fe_val_vel.get_quadrature(),
                    data.fe_val_vel.get_update_flags())
                , fe_val_pres(data.fe_val_pres.get_fe(),
                    data.fe_val_pres.get_quadrature(),
                    data.fe_val_pres.get_update_flags())
                , fe_val_def_grad(data.fe_val_def_grad.get_fe(),
                    data.fe_val_def_grad.get_quadrature(),
                    data.fe_val_def_grad.get_update_flags())
            {}
        };

        void assemble_one_cell_of_gradient(const IteratorPair& SI,
            InitGradScratchData& scratch,
            InitGradPerTaskData& data);

        void copy_gradient_local_to_global(const InitGradPerTaskData& data);

        void assemble_pres_Lap_term();

        struct pres_Lap_PerTaskData
        {
            FullMatrix<double>                   local_pres_Lap;
            std::vector<types::global_dof_index> local_dof_indices;
            AdvectionPerTaskData(const unsigned int dpc)
                : local_pres_Lap(dpc, dpc)
                , local_dof_indices(dpc)
            {}
        };

        struct pres_Lap_ScratchData
        {
            unsigned int                nqp;
            unsigned int                dpc;
            std::vector<Point<dim>>     u_star_local;
            std::vector<Tensor<1, dim>> grad_u_star;
            std::vector<double>         u_star_tmp;
            FEValues<dim>               fe_val;
            pres_Lap_ScratchData(const FE_Q<dim>& fe,
                const QGauss<dim>& quad,
                const UpdateFlags  flags)
                : nqp(quad.size())
                , dpc(fe.n_dofs_per_cell())
                , u_star_local(nqp)
                , grad_u_star(nqp)
                , u_star_tmp(nqp)
                , fe_val(fe, quad, flags)
            {}

            pres_Lap_ScratchData(const AdvectionScratchData& data)
                : nqp(data.nqp)
                , dpc(data.dpc)
                , u_star_local(nqp)
                , grad_u_star(nqp)
                , u_star_tmp(nqp)
                , fe_val(data.fe_val.get_fe(),
                    data.fe_val.get_quadrature(),
                    data.fe_val.get_update_flags())
            {}
        };

        void assemble_one_cell_of_pres_Lap(
            const typename DoFHandler<dim>::active_cell_iterator& cell,
            AdvectionScratchData& scratch,
            AdvectionPerTaskData& data);

        void copy_pres_Lap_local_to_global(const AdvectionPerTaskData& data);
        
        void assemble_advection_term();

        struct AdvectionPerTaskData
        {
            FullMatrix<double>                   local_advection;
            std::vector<types::global_dof_index> local_dof_indices;
            AdvectionPerTaskData(const unsigned int dpc)
                : local_advection(dpc, dpc)
                , local_dof_indices(dpc)
            {}
        };

        struct AdvectionScratchData
        {
            unsigned int                nqp;
            unsigned int                dpc;
            std::vector<Point<dim>>     u_star_local;
            std::vector<Tensor<1, dim>> grad_u_star;
            std::vector<double>         u_star_tmp;
            FEValues<dim>               fe_val;
            AdvectionScratchData(const FE_Q<dim>& fe,
                const QGauss<dim>& quad,
                const UpdateFlags  flags)
                : nqp(quad.size())
                , dpc(fe.n_dofs_per_cell())
                , u_star_local(nqp)
                , grad_u_star(nqp)
                , u_star_tmp(nqp)
                , fe_val(fe, quad, flags)
            {}

            AdvectionScratchData(const AdvectionScratchData& data)
                : nqp(data.nqp)
                , dpc(data.dpc)
                , u_star_local(nqp)
                , grad_u_star(nqp)
                , u_star_tmp(nqp)
                , fe_val(data.fe_val.get_fe(),
                    data.fe_val.get_quadrature(),
                    data.fe_val.get_update_flags())
            {}
        };

        void assemble_one_cell_of_advection(
            const typename DoFHandler<dim>::active_cell_iterator& cell,
            AdvectionScratchData& scratch,
            AdvectionPerTaskData& data);

        void copy_advection_local_to_global(const AdvectionPerTaskData& data);
        void diffusion_component_solve(const unsigned int d);

        void output_results(const unsigned int step);

    };

    template <int dim> 
    NavierStokesProjection<dim>::NavierStokesProjection(
        const RunTimeParameters::Data_Storage& data)
        : type(data.form)
        , deg(data.pressure_degree)
        , dt(data.dt)
        , t_0(data.initial_time)
        , T(data.final_time)
        , Re(data.Reynolds)
        , vel_exact(data.initial_time)
        , fe_momentum(deg + 1)
        , fe_pressure(deg)
        , fe_def_grad(deg)
        , dof_handler_momentum(triangulation)
        , dof_handler_pressure(triangulation)
        , dof_handler_def_grad(triangulation)
        , quadrature_pressure(deg + 1)
        , quadrature_momentum(deg + 2)
        , quadrature_def_grad(def+1)
        , vel_max_its(data.vel_max_iterations)
        , vel_Krylov_size(data.vel_Krylov_size)
        , vel_off_diagonals(data.vel_off_diagonals)
        , vel_update_prec(data.vel_update_prec)
        , vel_eps(data.vel_eps)
        , vel_diag_strength(data.vel_diag_strength)
    {
        if (deg < 1)
            std::cout
            << " WARNING: The chosen pair of finite element spaces is not stable."
            << std::endl
            << " The obtained results will be nonsense" << std::endl;

        AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

        create_triangulation_and_dofs(parameters.n_ref);
        initialize();
    }
    template <int dim>
    void NavierStokesProjection<dim>::create_triangulation_and_dofs(
        const unsigned int n_refines)
    {
        std::vector<Point<2>> vertices = {
            {0.0,0.0} , {0.0,0.44}, {0.48, 0.6}, {0.48, 0.44} };

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
                    if (face_center[0] == 0) {
                        face->set_boundary_id(4);
                    }
                    if (abs(face_center[0] - 0.48) < 0.001) {
                        face->set_boundary_id(5);
                    }
                }
        cout << triangulation.n_global_levels() << std::endl;

        std::cout << "Number of refines = " << n_refines << std::endl;
        triangulation.refine_global(n_refines);
        std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

        boundary_ids = triangulation.get_boundary_ids();

        dof_handler_momentum.distribute_dofs(fe_momentum);
        DoFRenumbering::boost::Cuthill_McKee(dof_handler_momentum);
        dof_handler_pressure.distribute_dofs(fe_pressure);
        DoFRenumbering::boost::Cuthill_McKee(dof_handler_pressure);
        dof_handler_def_grad.distribute_dofs(fe_def_grad);
        DoFRenumbering::boost::Cuthill_McKee(dof_handler_def_grad);

        initialize_momentum_matrices();
        initialize_pressure_matrices();
        initialize_def_grad_matrices();

        initialize_gradient_operator();

        pres_n.reinit(dof_handler_pressure.n_dofs());
        pres_n_minus_1.reinit(dof_handler_pressure.n_dofs());
        pres_tmp.reinit(dof_handler_pressure.n_dofs());
        for (unsigned int d = 0; d < dim; ++d)
        {
            u_n[d].reinit(dof_handler_momentum.n_dofs());
            u_n_minus_1[d].reinit(dof_handler_momentum.n_dofs());
            u_star[d].reinit(dof_handler_momentum.n_dofs());
            force[d].reinit(dof_handler_momentum.n_dofs());
            u_RHS[d].reinit(dof_handler_momentum.n_dofs());
            u_RHS_2[d].reinit(dof_handler_momentum.n_dofs());
        }
        
        u_tmp.reinit(dof_handler_momentum.n_dofs());

        for (unsigned int d = 0; d< dim * dim; ++d)
        {
            FF_n[d].reinit(dof_handler_def_grad.n_dofs());
            FF_star[d].reinit(dof_handler_def_grad.n_dofs());
            FF_n_minus_1[d].reinit(dof_handler_def_grad.n_dofs());
            FF_RHS[d].reinit(dof_handler_def_grad.n_dofs());

        }

        std::cout << "dim (X_h) = " << (dof_handler_momentum.n_dofs() * dim)
            << std::endl
            << "dim (M_h) = " << dof_handler_pressure.n_dofs()
            << std::endl
            << "Re        = " << Re << std::endl
            << std::endl;
    }

    template <int dim>
    void NavierStokesProjection<dim>::initialize()
    {
        vel_Laplace_plus_Mass = 0.;
        vel_Laplace_plus_Mass.add(1. / Re, vel_Laplace);
        vel_Laplace_plus_Mass.add(1.5 / dt, vel_Mass);

        EquationData::Pressure<dim> pres(t_0);
        VectorTools::interpolate(dof_handler_pressure, pres, pres_n_minus_1);
        pres.advance_time(dt);
        VectorTools::interpolate(dof_handler_pressure, pres, pres_n);
        phi_n = 0.;
        phi_n_minus_1 = 0.;
        for (unsigned int d = 0; d < dim; ++d)
        {
            vel_exact.set_time(t_0);
            vel_exact.set_component(d);
            VectorTools::interpolate(dof_handler_momentum,
                vel_exact,
                u_n_minus_1[d]);
            vel_exact.advance_time(dt);
            VectorTools::interpolate(dof_handler_momentum, vel_exact, u_n[d]);
        }
    }

    template <int dim>
    void NavierStokesProjection<dim>::initialize_momentum_matrices()
    {
        {
            DynamicSparsityPattern dsp(dof_handler_momentum.n_dofs(),
                dof_handler_momentum.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler_momentum, dsp);
            sparsity_pattern_momentum.copy_from(dsp);
        }

        
        vel_Mass.reinit(sparsity_pattern_momentum);

        MatrixCreator::create_mass_matrix(dof_handler_momentum,
            quadrature_momentum,
            vel_Mass);

    }

    template <int dim>
    void NavierStokesProjection<dim>::initialize_pressure_matrices()
    {
        {
            DynamicSparsityPattern dsp(dof_handler_pressure.n_dofs(),
                dof_handler_pressure.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp);
            sparsity_pattern_pressure.copy_from(dsp);
        }

        pres_Laplace.reinit(sparsity_pattern_pressure);
        pres_iterative.reinit(sparsity_pattern_pressure);
        pres_Mass.reinit(sparsity_pattern_pressure);

        MatrixCreator::create_mass_matrix(dof_handler_pressure,
            quadrature_pressure,
            pres_Mass);
    }

template <int dim>
void NavierStokesProjection<dim>::initialize_def_grad_matrices()
{
    {
        DynamicSparsityPattern dsp(dof_handler_def_grad.n_dofs(),
            dof_handler_def_grad.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_def_grad, dsp);
        sparsity_pattern_def_grad.copy_from(dsp);
    }

    
    def_grad_Mass.reinit(sparsity_pattern_def_grad);


    MatrixCreator::create_mass_matrix(dof_handler_def_grad,
        quadrature_def_grad,
        def_grad_Mass);
}

    template <int dim>
    void NavierStokesProjection<dim>::initialize_gradient_operator()
    {
        {
            DynamicSparsityPattern dsp(dof_handler_momentum.n_dofs(),
                dof_handler_pressure.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler_momentum,
                dof_handler_pressure,
                dsp);
            sparsity_pattern_pres_vel.copy_from(dsp);
        }

        InitGradPerTaskData per_task_data(0,
            fe_momentum.n_dofs_per_cell(),
            fe_pressure.n_dofs_per_cell(),
            fe_def_grad.n_dofs_per_cell());
        InitGradScratchData scratch_data(fe_momentum,
            fe_pressure,
            quadrature_momentum,
            update_gradients | update_JxW_values,
            update_values);

        for (unsigned int d = 0; d < dim; ++d)
        {
            pres_Diff[d].reinit(sparsity_pattern_pres_vel);
            per_task_data.d = d;
            WorkStream::run(
                IteratorPair(IteratorTuple(dof_handler_momentum.begin_active(),
                    dof_handler_pressure.begin_active())),
                IteratorPair(IteratorTuple(dof_handler_momentum.end(),
                    dof_handler_pressure.end())),
                *this,
                &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
                &NavierStokesProjection<dim>::copy_gradient_local_to_global,
                scratch_data,
                per_task_data);
        }
    }

    template <int dim>
    void NavierStokesProjection<dim>::assemble_one_cell_of_gradient(
        const IteratorPair& SI,
        InitGradScratchData& scratch,
        InitGradPerTaskData& data)
    {
        scratch.fe_val_vel.reinit(std::get<0>(*SI));
        scratch.fe_val_pres.reinit(std::get<1>(*SI));

        std::get<0>(*SI)->get_dof_indices(data.vel_local_dof_indices);
        std::get<1>(*SI)->get_dof_indices(data.pres_local_dof_indices);

        data.local_grad = 0.;
        for (unsigned int q = 0; q < scratch.nqp; ++q)
        {
            for (unsigned int i = 0; i < data.vel_dpc; ++i)
                for (unsigned int j = 0; j < data.pres_dpc; ++j)
                    data.local_grad(i, j) +=
                    -scratch.fe_val_vel.JxW(q) *
                    scratch.fe_val_vel.shape_grad(i, q)[data.d] *
                    scratch.fe_val_pres.shape_value(j, q);
        }
    }


    template <int dim>
    void NavierStokesProjection<dim>::copy_gradient_local_to_global(
        const InitGradPerTaskData& data)
    {
        for (unsigned int i = 0; i < data.vel_dpc; ++i)
            for (unsigned int j = 0; j < data.pres_dpc; ++j)
                pres_Diff[data.d].add(data.vel_local_dof_indices[i],
                    data.pres_local_dof_indices[j],
                    data.local_grad(i, j));
    }

    template <int dim>
    void NavierStokesProjection<dim>::run(const bool         verbose,
        const unsigned int output_interval)
    {
        ConditionalOStream verbose_cout(std::cout, verbose);

        const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);
        vel_exact.set_time(2. * dt);
        output_results(1);
        for (unsigned int n = 2; n <= n_steps; ++n)
        {
            if (n % output_interval == 0)
            {
                verbose_cout << "Plotting Solution" << std::endl;
                output_results(n);
            }
            std::cout << "Step = " << n << " Time = " << (n * dt) << std::endl;
            verbose_cout << "  Interpolating the momentum " << std::endl;

            interpolate_momentum();
            verbose_cout << "  Diffusion Step" << std::endl;
            if (n % vel_update_prec == 0)
                verbose_cout << "    With reinitialization of the preconditioner"
                << std::endl;
            diffusion_step((n % vel_update_prec == 0) || (n == 2));
            verbose_cout << "  Projection Step" << std::endl;
            projection_step((n == 2));
            verbose_cout << "  Updating the Pressure" << std::endl;
            update_pressure((n == 2));
            vel_exact.advance_time(dt);
        }
        output_results(n_steps);
    }



    template <int dim>
    void NavierStokesProjection<dim>::interpolate_momentum()
    {
        for (unsigned int d = 0; d < dim; ++d)
        {
            u_star[d].equ(2., u_n[d]);
            u_star[d] -= u_n_minus_1[d];
        }
    }

    template <int dim>
    void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
    {
        pres_tmp.equ(-1., pres_n);
        pres_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_minus_1);

        assemble_advection_term();

        for (unsigned int d = 0; d < dim; ++d)
        {
            force[d] = 0.;
            u_tmp.equ(2. / dt, u_n[d]);
            u_tmp.add(-.5 / dt, u_n_minus_1[d]);
            vel_Mass.vmult_add(force[d], u_tmp);

            pres_Diff[d].vmult_add(force[d], pres_tmp);
            u_n_minus_1[d] = u_n[d];

            vel_it_matrix[d].copy_from(vel_Laplace_plus_Mass);
            vel_it_matrix[d].add(1., vel_Advection);

            vel_exact.set_component(d);
            boundary_values.clear();
            for (const auto& boundary_id : boundary_ids)
            {
                switch (boundary_id)
                {
                case 1:
                    VectorTools::interpolate_boundary_values(
                        dof_handler_momentum,
                        boundary_id,
                        Functions::ZeroFunction<dim>(),
                        boundary_values);
                    break;
                case 2:
                    VectorTools::interpolate_boundary_values(dof_handler_momentum,
                        boundary_id,
                        vel_exact,
                        boundary_values);
                    break;
                case 3:
                    if (d != 0)
                        VectorTools::interpolate_boundary_values(
                            dof_handler_momentum,
                            boundary_id,
                            Functions::ZeroFunction<dim>(),
                            boundary_values);
                    break;
                case 4:
                    VectorTools::interpolate_boundary_values(
                        dof_handler_momentum,
                        boundary_id,
                        Functions::ZeroFunction<dim>(),
                        boundary_values);
                    break;
                default:
                    Assert(false, ExcNotImplemented());
                }
            }
            MatrixTools::apply_boundary_values(boundary_values,
                vel_it_matrix[d],
                u_n[d],
                force[d]);
        }


        Threads::TaskGroup<void> tasks;
        for (unsigned int d = 0; d < dim; ++d)
        {
            if (reinit_prec)
                prec_momentum[d].initialize(vel_it_matrix[d],
                    SparseILU<double>::AdditionalData(
                        vel_diag_strength, vel_off_diagonals));
            tasks += Threads::new_task(
                &NavierStokesProjection<dim>::diffusion_component_solve, *this, d);
        }
        tasks.join_all();
    }



    template <int dim>
    void
        NavierStokesProjection<dim>::diffusion_component_solve(const unsigned int d)
    {
        SolverControl solver_control(vel_max_its, vel_eps * force[d].l2_norm());
        SolverGMRES<Vector<double>> gmres(
            solver_control,
            SolverGMRES<Vector<double>>::AdditionalData(vel_Krylov_size));
        gmres.solve(vel_it_matrix[d], u_n[d], force[d], prec_momentum[d]);
    }

    template <int dim>
    void NavierStokesProjection<dim>::assemble_advection_term()
    {
        vel_Advection = 0.;
        AdvectionPerTaskData data(fe_momentum.n_dofs_per_cell());
        AdvectionScratchData scratch(fe_momentum,
            quadrature_momentum,
            update_values | update_JxW_values |
            update_gradients);
        WorkStream::run(
            dof_handler_momentum.begin_active(),
            dof_handler_momentum.end(),
            *this,
            &NavierStokesProjection<dim>::assemble_one_cell_of_advection,
            &NavierStokesProjection<dim>::copy_advection_local_to_global,
            scratch,
            data);
    }



    template <int dim>
    void NavierStokesProjection<dim>::assemble_one_cell_of_advection(
        const typename DoFHandler<dim>::active_cell_iterator& cell,
        AdvectionScratchData& scratch,
        AdvectionPerTaskData& data)
    {
        scratch.fe_val.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);
        for (unsigned int d = 0; d < dim; ++d)
        {
            scratch.fe_val.get_function_values(u_star[d], scratch.u_star_tmp);
            for (unsigned int q = 0; q < scratch.nqp; ++q)
                scratch.u_star_local[q](d) = scratch.u_star_tmp[q];
        }

        for (unsigned int d = 0; d < dim; ++d)
        {
            scratch.fe_val.get_function_gradients(u_star[d], scratch.grad_u_star);
            for (unsigned int q = 0; q < scratch.nqp; ++q)
            {
                if (d == 0)
                    scratch.u_star_tmp[q] = 0.;
                scratch.u_star_tmp[q] += scratch.grad_u_star[q][d];
            }
        }

        data.local_advection = 0.;
        for (unsigned int q = 0; q < scratch.nqp; ++q)
            for (unsigned int i = 0; i < scratch.dpc; ++i)
                for (unsigned int j = 0; j < scratch.dpc; ++j)
                    data.local_advection(i, j) += (scratch.u_star_local[q] *
                        scratch.fe_val.shape_grad(j, q) *
                        scratch.fe_val.shape_value(i, q)
                        +
                        0.5 *
                        scratch.u_star_tmp[q] *
                        scratch.fe_val.shape_value(i, q) *
                        scratch.fe_val.shape_value(j, q))
                    * scratch.fe_val.JxW(q);
    }



    template <int dim>
    void NavierStokesProjection<dim>::copy_advection_local_to_global(
        const AdvectionPerTaskData& data)
    {
        for (unsigned int i = 0; i < fe_momentum.n_dofs_per_cell(); ++i)
            for (unsigned int j = 0; j < fe_momentum.n_dofs_per_cell(); ++j)
                vel_Advection.add(data.local_dof_indices[i],
                    data.local_dof_indices[j],
                    data.local_advection(i, j));
    }

    template <int dim>
    void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
    {
        pres_iterative.copy_from(pres_Laplace);

        pres_tmp = 0.;
        for (unsigned d = 0; d < dim; ++d)
            pres_Diff[d].Tvmult_add(pres_tmp, u_n[d]);

        phi_n_minus_1 = phi_n;

        static std::map<types::global_dof_index, double> bval;
        if (reinit_prec)
            VectorTools::interpolate_boundary_values(dof_handler_pressure,
                3,
                Functions::ZeroFunction<dim>(),
                bval);

        MatrixTools::apply_boundary_values(bval, pres_iterative, phi_n, pres_tmp);

        if (reinit_prec)
            prec_pres_Laplace.initialize(pres_iterative,
                SparseILU<double>::AdditionalData(
                    vel_diag_strength, vel_off_diagonals));

        SolverControl solvercontrol(vel_max_its, vel_eps * pres_tmp.l2_norm());
        SolverCG<Vector<double>> cg(solvercontrol);
        cg.solve(pres_iterative, phi_n, pres_tmp, prec_pres_Laplace);

        phi_n *= 1.5 / dt;
    }

    template <int dim>
    void NavierStokesProjection<dim>::update_pressure(const bool reinit_prec)
    {
        pres_n_minus_1 = pres_n;
        switch (type)
        {
        case RunTimeParameters::Method::standard:
            pres_n += phi_n;
            break;
        case RunTimeParameters::Method::rotational:
            if (reinit_prec)
                prec_mass.initialize(pres_Mass);
            pres_n = pres_tmp;
            prec_mass.solve(pres_n);
            pres_n.sadd(1. / Re, 1., pres_n_minus_1);
            pres_n += phi_n;
            break;
        default:
            Assert(false, ExcNotImplemented());
        };
    }

    template <int dim>
    void NavierStokesProjection<dim>::output_results(const unsigned int step)
    {
        assemble_vorticity((step == 1));
        const FESystem<dim> joint_fe(
            fe_momentum, dim, fe_pressure, 1, fe_momentum, 1);
        DoFHandler<dim> joint_dof_handler(triangulation);
        joint_dof_handler.distribute_dofs(joint_fe);
        Assert(joint_dof_handler.n_dofs() ==
            ((dim + 1) * dof_handler_momentum.n_dofs() +
                dof_handler_pressure.n_dofs()),
            ExcInternalError());
        Vector<double> joint_solution(joint_dof_handler.n_dofs());
        std::vector<types::global_dof_index> loc_joint_dof_indices(
            joint_fe.n_dofs_per_cell()),
            loc_vel_dof_indices(fe_momentum.n_dofs_per_cell()),
            loc_pres_dof_indices(fe_pressure.n_dofs_per_cell());
        typename DoFHandler<dim>::active_cell_iterator
            joint_cell = joint_dof_handler.begin_active(),
            joint_endc = joint_dof_handler.end(),
            vel_cell = dof_handler_momentum.begin_active(),
            pres_cell = dof_handler_pressure.begin_active();
        for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
        {
            joint_cell->get_dof_indices(loc_joint_dof_indices);
            vel_cell->get_dof_indices(loc_vel_dof_indices);
            pres_cell->get_dof_indices(loc_pres_dof_indices);
            for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i)
                switch (joint_fe.system_to_base_index(i).first.first)
                {
                case 0:
                    Assert(joint_fe.system_to_base_index(i).first.second < dim,
                        ExcInternalError());
                    joint_solution(loc_joint_dof_indices[i]) =
                        u_n[joint_fe.system_to_base_index(i).first.second](
                            loc_vel_dof_indices[joint_fe.system_to_base_index(i)
                            .second]);
                    break;
                case 1:
                    Assert(joint_fe.system_to_base_index(i).first.second == 0,
                        ExcInternalError());
                    joint_solution(loc_joint_dof_indices[i]) =
                        pres_n(loc_pres_dof_indices[joint_fe.system_to_base_index(i)
                            .second]);
                    break;
                case 2:
                    Assert(joint_fe.system_to_base_index(i).first.second == 0,
                        ExcInternalError());
                    joint_solution(loc_joint_dof_indices[i]) = rot_u(
                        loc_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
                    break;
                default:
                    Assert(false, ExcInternalError());
                }
        }
        std::vector<std::string> joint_solution_names(dim, "v");
        joint_solution_names.emplace_back("p");
        joint_solution_names.emplace_back("rot_u");
        DataOut<dim> data_out;
        data_out.attach_dof_handler(joint_dof_handler);
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            component_interpretation(
                dim + 2, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] =
            DataComponentInterpretation::component_is_scalar;
        component_interpretation[dim + 1] =
            DataComponentInterpretation::component_is_scalar;
        data_out.add_data_vector(joint_solution,
            joint_solution_names,
            DataOut<dim>::type_dof_data,
            component_interpretation);
        data_out.build_patches(deg + 1);
        std::ofstream output("solution-" + Utilities::int_to_string(step, 5) +
            ".vtk");
        data_out.write_vtk(output);
    }

    template <int dim>
    void NavierStokesProjection<dim>::assemble_vorticity(const bool reinit_prec)
    {
        Assert(dim == 2, ExcNotImplemented());
        if (reinit_prec)
            prec_vel_mass.initialize(vel_Mass);

        FEValues<dim>      fe_val_vel(fe_momentum,
            quadrature_momentum,
            update_gradients | update_JxW_values |
            update_values);
        const unsigned int dpc = fe_momentum.n_dofs_per_cell(),
            nqp = quadrature_momentum.size();
        std::vector<types::global_dof_index> ldi(dpc);
        Vector<double>                       loc_rot(dpc);

        std::vector<Tensor<1, dim>> grad_u1(nqp), grad_u2(nqp);
        rot_u = 0.;

        for (const auto& cell : dof_handler_momentum.active_cell_iterators())
        {
            fe_val_vel.reinit(cell);
            cell->get_dof_indices(ldi);
            fe_val_vel.get_function_gradients(u_n[0], grad_u1);
            fe_val_vel.get_function_gradients(u_n[1], grad_u2);
            loc_rot = 0.;
            for (unsigned int q = 0; q < nqp; ++q)
                for (unsigned int i = 0; i < dpc; ++i)
                    loc_rot(i) += (grad_u2[q][0] - grad_u1[q][1]) *
                    fe_val_vel.shape_value(i, q) *
                    fe_val_vel.JxW(q);

            for (unsigned int i = 0; i < dpc; ++i)
                rot_u(ldi[i]) += loc_rot(i);
        }

        prec_vel_mass.solve(rot_u);
    }
} // namespace NonlinearElasticity

int main()
{
    try
    {
        using namespace NonlinearElasticity;

        RunTimeParameters::Data_Storage data;
        data.read_data("parameter-file.prm");

        deallog.depth_console(data.verbose ? 2 : 0);

        NavierStokesProjection<2> test(data);
        test.run(data.verbose, data.output_interval);
    }
    catch (std::exception& exc)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }
    std::cout << "----------------------------------------------------"
        << std::endl
        << "Apparently everything went fine!" << std::endl
        << "Don't forget to brush your teeth :-)" << std::endl
        << std::endl;
    return 0;
}
