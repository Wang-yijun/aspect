/*
  Copyright (C) 2019 - 2020 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_material_model_LPO_AV_3D_h
#define _aspect_material_model_LPO_AV_3D_h

#include <aspect/material_model/interface.h>
#include <aspect/material_model/utilities.h>
#include <aspect/material_model/simple.h>
#include <aspect/material_model/rheology/diffusion_dislocation.h>
#include <aspect/material_model/rheology/diffusion_creep.h>
#include <aspect/material_model/rheology/dislocation_creep.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator/assemblers/interface.h>
#include <aspect/material_model/equation_of_state/interface.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * A material model that implements the micromechanical behaviour of olivine grains to create anisotropic viscosity.
     * Based on Hansen et al., 2016 (JGR) and Kiraly et al., 2020 (G3).
     * The micromechanical model requires the euler angles of the olivine grains (now stored on 3 compositional field),
     * the grainsize, tempreature, and strain rate to calculate the stress that is needed to create the input strain rate.
     * The material model otherwise is based on the Simple material model.
     * @ingroup MaterialModels
     */
    template <int dim>
    class AV : public NamedAdditionalMaterialOutputs<dim>
    {
      public:
        AV(const unsigned int n_points);

        std::vector<double> get_nth_output(const unsigned int idx) const override;

        /**
         * Stress-strain "director" tensors at the given positions. This
         * variable is used to implement anisotropic viscosity.
         *
         * @note The strain rate term in equation (1) of the manual will be
         * multiplied by this tensor *and* the viscosity scalar ($\eta$ /i.e. effective viscosity), as
         * described in the manual section titled "Constitutive laws". This
         * variable is assigned the rank-four identity tensor by default.
         * This leaves the isotropic constitutive law unchanged if the material
         * model does not explicitly assign a value.
         */
        std::vector<SymmetricTensor<4,dim>> stress_strain_directors;



    };

    /*template <int matrix_size>
    void check_eigenvalues_positive(const SymmetricTensor<2,matrix_size> &matrix);*/


    template <int dim>
    class LPO_AV_3D : public MaterialModel::Simple<dim>
    {
      public:
        void initialize() override;
        void evaluate (const MaterialModel::MaterialModelInputs<dim> &in,
                       MaterialModel::MaterialModelOutputs<dim> &out) const override;
        static void declare_parameters (ParameterHandler &prm);
        bool is_compressible () const override;
        double reference_viscosity () const;
        void create_additional_named_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const override;
        /**
         * Read the parameters this class declares from the parameter file.
         * If @p expected_n_phases_per_composition points to a vector of
         * unsigned integers this is considered the number of phases
         * for each compositional field and will be checked against the parsed
         * parameters.
         */
        void
        parse_parameters (ParameterHandler &prm,
                          const std::unique_ptr<std::vector<unsigned int>> &expected_n_phases_per_composition = nullptr);

        /**
         * Compute the viscosity based on the composite viscous creep law.
         * If @p n_phase_transitions_per_composition points to a vector of
         * unsigned integers this is considered the number of phase transitions
         * for each compositional field and viscosity will be first computed on
         * each phase and then averaged for each compositional field.
         */
        std::vector<double>
        compute_diffusion_parameters (const double strain_rate,
                                      const double pressure,
                                      const double temperature,
                                      const unsigned int composition,
                                      const std::vector<double> &phase_function_values = std::vector<double>(),
                                      const std::vector<unsigned int> &n_phase_transitions_per_composition = std::vector<unsigned int>()) const;

      protected:
        /**
         * Object that handles phase transitions.
         * Allows it to compute the phase function for each individual phase
         * transition in the model, given the temperature, pressure, depth,
         * and density gradient.
         */
        MaterialUtilities::PhaseFunction<dim> phase_function;
      private:
        /**
         * Object for computing viscous creep viscosities.
         */
        Rheology::DiffusionDislocation<dim> diffusion_dislocation;
        Rheology::DiffusionCreep<dim> diffusion_creep;
        Rheology::DislocationCreep<dim> dislocation_creep;

        double reference_T;

        double thermal_diffusivity;
        double heat_capacity;

        std::vector<double> densities;
        std::vector<double> thermal_expansivities;
        
        double eta; //reference viscosity
        /**
         * Defining a minimum strain rate stabilizes the viscosity calculation,
         * which involves a division by the strain rate. Units: 1/s.
         */
        double min_strain_rate;
        std::vector<unsigned int> cpo_bingham_avg_a, cpo_bingham_avg_b, cpo_bingham_avg_c;
        double grain_size;
        std::vector<double> CnI_F, CnI_G, CnI_H, CnI_L, CnI_M, CnI_N;

        unsigned int n_grains;
        EquationOfState::LinearizedIncompressible<dim> equation_of_state;
        void set_assemblers(const SimulatorAccess<dim> &,
                            Assemblers::Manager<dim> &assemblers) const;

    };
  }
}

#endif
