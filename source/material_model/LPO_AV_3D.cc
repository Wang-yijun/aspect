
/*
 Copyright (C) 2015 - 2020 by the authors of the ASPECT code.

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

#include <aspect/material_model/LPO_AV_3D.h>
#include <aspect/material_model/equation_of_state/interface.h>
#include <aspect/introspection.h>
#include <aspect/material_model/interface.h>
#include <aspect/plugins.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

//#include <world_builder/utilities.h>

#include <aspect/simulator_access.h>
#include <aspect/simulator.h>
#include <aspect/global.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/geometry_info.h>
#include <aspect/simulator_access.h>

#include <aspect/material_model/simple.h>
#include <aspect/material_model/grain_size.h>
#include <aspect/heating_model/shear_heating.h>
#include <aspect/heating_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/simulator/assemblers/stokes.h>
#include <aspect/simulator_signals.h>
#include <aspect/postprocess/particles.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <boost/random.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
//#include <aspect/postprocess/particle_lpo.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/signaling_nan.h>


namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * Additional output fields for anisotropic viscosities to be added to
     * the MaterialModel::MaterialModelOutputs structure and filled in the
     * MaterialModel::Interface::evaluate() function.
     */


    namespace
    {

      template <int dim>
      std::vector<std::string> make_AV_additional_outputs_names()
      {
        std::vector<std::string> names;

        for (unsigned int i = 0; i < Tensor<4,dim>::n_independent_components ; ++i)
          {
            TableIndices<4> indices(Tensor<4,dim>::unrolled_to_component_indices(i));
            names.push_back("anisotropic_viscosity"+std::to_string(indices[0]+1)+std::to_string(indices[1]+1)+std::to_string(indices[2]+1)+std::to_string(indices[3]+1));
          }
        return names;
      }
    }



    template <int dim>
    AV<dim>::AV (const unsigned int n_points)
      :
      NamedAdditionalMaterialOutputs<dim>(make_AV_additional_outputs_names<dim>()),
      stress_strain_directors(n_points, dealii::identity_tensor<dim> ())
    {}



    template <int dim>
    std::vector<double>
    AV<dim>::get_nth_output(const unsigned int idx) const
    {
      std::vector<double> output(stress_strain_directors.size());
      for (unsigned int i = 0; i < stress_strain_directors.size() ; ++i)
        {
          output[i]= stress_strain_directors[i][Tensor<4,dim>::unrolled_to_component_indices(idx)];
        }
      return output;
    }
  }
}

namespace aspect
{
  namespace Assemblers
  {
    /**
     * A class containing the functions to assemble the Stokes preconditioner.
     */
    template <int dim>
    class StokesPreconditionerAV : public Assemblers::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        virtual
        void
        execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch,
                internal::Assembly::CopyData::CopyDataBase<dim> &data) const;

        /**
         * Create AnisotropicViscosities.
         */
        virtual void create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const;
    };

    /**
     * This class assembles the terms for the matrix and right-hand-side of the incompressible
     * Stokes equation for the current cell.
     */
    template <int dim>
    class StokesIncompressibleTermsAV : public Assemblers::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        virtual
        void
        execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch,
                internal::Assembly::CopyData::CopyDataBase<dim> &data) const;

        /**
         * Create AdditionalMaterialOutputsStokesRHS if we need to do so.
         */
        virtual void create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const;
    };



    template <int dim>
    void
    StokesPreconditionerAV<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesPreconditioner<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesPreconditioner<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesPreconditioner<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesPreconditioner<dim>& > (data_base);

      const MaterialModel::AV<dim> *anisotropic_viscosity =
        scratch.material_model_outputs.template get_additional_output<MaterialModel::AV<dim>>();

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points           = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      // First loop over all dofs and find those that are in the Stokes system
      // save the component (pressure and dim velocities) each belongs to.
      for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
        {
          if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
            {
              scratch.dof_component_indices[i_stokes] = fe.system_to_component_index(i).first;
              ++i_stokes;
            }
          ++i;
        }

      // Loop over all quadrature points and assemble their contributions to
      // the preconditioner matrix
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0, i_stokes = 0; i_stokes < stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.grads_phi_u[i_stokes] =
                    scratch.finite_element_values[introspection.extractors
                                                  .velocities].symmetric_gradient(i, q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection
                                                                          .extractors.pressure].value(i, q);
                  ++i_stokes;
                }
              ++i;
            }

          const double eta = scratch.material_model_outputs.viscosities[q];

          //std::cout <<"The effective viscosity is: ";
          //std::cout << eta <<std::endl;
          const double one_over_eta = 1. / eta;
          const SymmetricTensor<4, dim> &stress_strain_director = anisotropic_viscosity->stress_strain_directors[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
            for (unsigned int j = 0; j < stokes_dofs_per_cell; ++j)
              if (scratch.dof_component_indices[i] ==
                  scratch.dof_component_indices[j])
                data.local_matrix(i, j) += (2.0 * eta * (scratch.grads_phi_u[i]
                                                         * stress_strain_director
                                                         * scratch.grads_phi_u[j])
                                            + one_over_eta * pressure_scaling
                                            * pressure_scaling
                                            * (scratch.phi_p[i]
                                               * scratch.phi_p[j]))
                                           * JxW;
        }
    }



    template <int dim>
    void
    StokesPreconditionerAV<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.viscosities.size();

      if (outputs.template get_additional_output<MaterialModel::AV<dim>>() == nullptr)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::AV<dim>> (n_points));
        }
    }



    template <int dim>
    void
    StokesIncompressibleTermsAV<dim>::
    execute (internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
             internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      internal::Assembly::Scratch::StokesSystem<dim> &scratch = dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>& > (scratch_base);
      internal::Assembly::CopyData::StokesSystem<dim> &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>& > (data_base);

      const MaterialModel::AV<dim> *anisotropic_viscosity =
        scratch.material_model_outputs.template get_additional_output<MaterialModel::AV<dim>>();

      const Introspection<dim> &introspection = this->introspection();
      const FiniteElement<dim> &fe = this->get_fe();
      const unsigned int stokes_dofs_per_cell = data.local_dof_indices.size();
      const unsigned int n_q_points    = scratch.finite_element_values.n_quadrature_points;
      const double pressure_scaling = this->get_pressure_scaling();

      const MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>
      *force = scratch.material_model_outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>();

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int i=0, i_stokes=0; i_stokes<stokes_dofs_per_cell; /*increment at end of loop*/)
            {
              if (introspection.is_stokes_component(fe.system_to_component_index(i).first))
                {
                  scratch.phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].value (i,q);
                  scratch.phi_p[i_stokes] = scratch.finite_element_values[introspection.extractors.pressure].value (i, q);
                  if (scratch.rebuild_stokes_matrix)
                    {
                      scratch.grads_phi_u[i_stokes] = scratch.finite_element_values[introspection.extractors.velocities].symmetric_gradient(i,q);
                      scratch.div_phi_u[i_stokes]   = scratch.finite_element_values[introspection.extractors.velocities].divergence (i, q);
                    }
                  ++i_stokes;
                }
              ++i;
            }
          // Viscosity scalar
          const double eta = (scratch.rebuild_stokes_matrix
                              ?
                              scratch.material_model_outputs.viscosities[q]
                              :
                              numbers::signaling_nan<double>());

          const SymmetricTensor<4, dim> &stress_strain_director = anisotropic_viscosity->stress_strain_directors[q];
          //std::cout << "director: " << stress_strain_director << std::endl;

          const Tensor<1,dim>
          gravity = this->get_gravity_model().gravity_vector (scratch.finite_element_values.quadrature_point(q));

          const double density = scratch.material_model_outputs.densities[q];
          const double JxW = scratch.finite_element_values.JxW(q);

          for (unsigned int i=0; i<stokes_dofs_per_cell; ++i)
            {
              data.local_rhs(i) += (density * gravity * scratch.phi_u[i])
                                   * JxW;

              if (force != nullptr)
                data.local_rhs(i) += (force->rhs_u[q] * scratch.phi_u[i]
                                      + pressure_scaling * force->rhs_p[q] * scratch.phi_p[i])
                                     * JxW;

              if (scratch.rebuild_stokes_matrix)
                for (unsigned int j=0; j<stokes_dofs_per_cell; ++j)
                  {
                    data.local_matrix(i,j) += ( eta * 2.0 * (scratch.grads_phi_u[i] * stress_strain_director * scratch.grads_phi_u[j])
                                                // assemble \nabla p as -(p, div v):
                                                - (pressure_scaling *
                                                   scratch.div_phi_u[i] * scratch.phi_p[j])
                                                // assemble the term -div(u) as -(div u, q).
                                                // Note the negative sign to make this
                                                // operator adjoint to the grad p term:
                                                - (pressure_scaling *
                                                   scratch.phi_p[i] * scratch.div_phi_u[j]))
                                              * JxW;
                  }
            }
        }
    }



    template <int dim>
    void
    StokesIncompressibleTermsAV<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const
    {
      const unsigned int n_points = outputs.viscosities.size();

      if (outputs.template get_additional_output<MaterialModel::AV<dim>>() == nullptr)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::AV<dim>> (n_points));
        }

      if (this->get_parameters().enable_additional_stokes_rhs
          && outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>() == nullptr)
        {
          outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>> (n_points));
        }
      Assert(!this->get_parameters().enable_additional_stokes_rhs
             ||
             outputs.template get_additional_output<MaterialModel::AdditionalMaterialOutputsStokesRHS<dim>>()->rhs_u.size()
             == n_points, ExcInternalError());
    }
  }

  namespace HeatingModel
  {
    template <int dim>
    class ShearHeatingAV : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Compute the heating model outputs for this class.
         */
        virtual
        void
        evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
                  const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
                  HeatingModel::HeatingModelOutputs &heating_model_outputs) const;

        /**
         * Allow the heating model to attach additional material model outputs.
         */
        virtual
        void
        create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &material_model_outputs) const;
    };



    template <int dim>
    void
    ShearHeatingAV<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
              const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      Assert(heating_model_outputs.heating_source_terms.size() == material_model_inputs.position.size(),
             ExcMessage ("Heating outputs need to have the same number of entries as the material model inputs."));

      Assert(heating_model_outputs.heating_source_terms.size() == material_model_inputs.strain_rate.size(),
             ExcMessage ("The shear heating plugin needs the strain rate!"));

      // Some material models provide dislocation viscosities and boundary area work fractions
      // as additional material outputs. If they are attached, use them.
      const MaterialModel::DislocationViscosityOutputs<dim> *disl_viscosity_out =
        material_model_outputs.template get_additional_output<MaterialModel::DislocationViscosityOutputs<dim>>();

      const MaterialModel::AV<dim> *anisotropic_viscosity =
        material_model_outputs.template get_additional_output<MaterialModel::AV<dim>>();

      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          // If there is an anisotropic viscosity, use it to compute the correct stress
          const SymmetricTensor<2,dim> &directed_strain_rate = ((anisotropic_viscosity != nullptr)
                                                                ?
                                                                anisotropic_viscosity->stress_strain_directors[q]
                                                                * material_model_inputs.strain_rate[q]
                                                                :
                                                                material_model_inputs.strain_rate[q]);

          const SymmetricTensor<2,dim> stress =
            2 * material_model_outputs.viscosities[q] *
            (this->get_material_model().is_compressible()
             ?
             directed_strain_rate - 1./3. * trace(directed_strain_rate) * unit_symmetric_tensor<dim>()
             :
             directed_strain_rate);

          const SymmetricTensor<2,dim> deviatoric_strain_rate =
            (this->get_material_model().is_compressible()
             ?
             material_model_inputs.strain_rate[q]
             - 1./3. * trace(material_model_inputs.strain_rate[q]) * unit_symmetric_tensor<dim>()
             :
             material_model_inputs.strain_rate[q]);

          heating_model_outputs.heating_source_terms[q] = stress * deviatoric_strain_rate;

          // If dislocation viscosities and boundary area work fractions are provided, reduce the
          // overall heating by this amount (which is assumed to increase surface energy)
          if (disl_viscosity_out != 0)
            {
              heating_model_outputs.heating_source_terms[q] *= 1 - disl_viscosity_out->boundary_area_change_work_fractions[q] *
                                                               material_model_outputs.viscosities[q] /disl_viscosity_out->dislocation_viscosities[q];
            }

          heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
        }
    }



    template <int dim>
    void
    ShearHeatingAV<dim>::
    create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &material_model_outputs) const
    {
      const unsigned int n_points = material_model_outputs.viscosities.size();

      if (material_model_outputs.template get_additional_output<MaterialModel::AV<dim>>() == nullptr)
        {
          material_model_outputs.additional_outputs.push_back(
            std::make_unique<MaterialModel::AV<dim>> (n_points));
        }

      this->get_material_model().create_additional_named_outputs(material_model_outputs);
    }
  }

}

namespace aspect
{

//Next session is a more evolved implementation of anisotropic viscosity in the material model based on Hansen et al 2016 and Kiraly et al 2020
  namespace MaterialModel
  {


    template <int dim>
    void
    LPO_AV_3D<dim>::set_assemblers(const SimulatorAccess<dim> &,
                                   Assemblers::Manager<dim> &assemblers) const
    {
      for (unsigned int i=0; i<assemblers.stokes_preconditioner.size(); ++i)
        {
          if (Plugins::plugin_type_matches<Assemblers::StokesPreconditioner<dim>>(*(assemblers.stokes_preconditioner[i])))
            assemblers.stokes_preconditioner[i] = std::make_unique<Assemblers::StokesPreconditionerAV<dim>> ();
        }

      for (unsigned int i=0; i<assemblers.stokes_system.size(); ++i)
        {
          if (Plugins::plugin_type_matches<Assemblers::StokesIncompressibleTerms<dim>>(*(assemblers.stokes_system[i])))
            assemblers.stokes_system[i] = std::make_unique<Assemblers::StokesIncompressibleTermsAV<dim>> ();
        }
    }



    template <int dim>
    void
    LPO_AV_3D<dim>::
    initialize()
    {
      this->get_signals().set_assemblers.connect (std::bind(&LPO_AV_3D<dim>::set_assemblers,
                                                            std::cref(*this),
                                                            std::placeholders::_1,
                                                            std::placeholders::_2));
      AssertThrow((dim==3),
                  ExcMessage("Olivine has 3 independent slip systems, allowing for deformation in 3 independent directions, hence these models only work in 3D"));

      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvector_a1"));
      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvector_a2"));
      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvector_a3"));
      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvalue_a1"));
      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvalue_a2"));
      lpo_bingham_avg_a.push_back (this->introspection().compositional_index_for_name("eigvalue_a3"));

      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvector_b1"));
      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvector_b2"));
      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvector_b3"));
      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvalue_b1"));
      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvalue_b2"));
      lpo_bingham_avg_b.push_back (this->introspection().compositional_index_for_name("eigvalue_b3"));

      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvector_c1"));
      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvector_c2"));
      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvector_c3"));
      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvalue_c1"));
      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvalue_c2"));
      lpo_bingham_avg_c.push_back (this->introspection().compositional_index_for_name("eigvalue_c3"));


    }




    template <int dim>
    double
    AV<dim>::J2_second_invariant(const SymmetricTensor<2,dim> t, const double min_strain_rate)
    {
      const double J2_strict = (1.0/6.0*(std::pow(double (t[0][0] - t[1][1]),2) + std::pow(double (t[1][1] - t[2][2]),2)+std::pow(double (t[2][2] - t[0][0]),2)))+(std::pow(t[0][1],2)+std::pow(t[1][2],2)+std::pow(t[2][0],2));
      const double J2 = std::max(J2_strict, std::pow(min_strain_rate,2)); //prevents having too small values (also used in compute_second_invariant for strain rate)
      return J2;
    }

    template<int dim>
    Tensor<2,3>
    AV<dim>::euler_angles_to_rotation_matrix(double phi1_d, double theta_d, double phi2_d)
    {
      constexpr double const_pi = 3.141592653589793238462643383279502884;
      const double degree_to_rad = const_pi/180.0;
      const double phi1 = phi1_d * degree_to_rad;
      const double theta = theta_d * degree_to_rad;
      const double phi2 = phi2_d * degree_to_rad;
      Tensor<2,3> rot_matrix;

      //R3*R2*R1 ZXZ rotation. Note it is not exactly the same as in utilities.cc
      rot_matrix[0][0] = cos(phi2)*cos(phi1) - cos(theta)*sin(phi1)*sin(phi2);
      rot_matrix[0][1] = -cos(phi2)*sin(phi1) - cos(theta)*cos(phi1)*sin(phi2);
      rot_matrix[0][2] = sin(phi2)*sin(theta);

      rot_matrix[1][0] = sin(phi2)*cos(phi1) + cos(theta)*sin(phi1)*cos(phi2);
      rot_matrix[1][1] = -sin(phi2)*sin(phi1) + cos(theta)*cos(phi1)*cos(phi2);
      rot_matrix[1][2] = -cos(phi2)*sin(theta);

      rot_matrix[2][0] = sin(theta)*sin(phi1);
      rot_matrix[2][1] = sin(theta)*cos(phi1);
      rot_matrix[2][2] = cos(theta);
      return rot_matrix;
    }



    template <>
    void
    LPO_AV_3D<2>::evaluate (const MaterialModel::MaterialModelInputs<2> &,
                            MaterialModel::MaterialModelOutputs<2> &) const
    {
      Assert (false, ExcNotImplemented());
    }


    template <>
    void
    LPO_AV_3D<3>::evaluate (const MaterialModel::MaterialModelInputs<3> &in,
                            MaterialModel::MaterialModelOutputs<3> &out) const
    {
      const int dim=3;

      MaterialModel::AV<dim> *anisotropic_viscosity;
      anisotropic_viscosity = out.template get_additional_output<MaterialModel::AV<dim>>();

      EquationOfStateOutputs<dim> eos_outputs (1);
      for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
        {
          //change these according to diffusion dislocation material model I guess
          equation_of_state.evaluate(in, q, eos_outputs);
          out.densities[q] = eos_outputs.densities[0];//Change this to 0 for the simple shear box test
          out.viscosities[q] = eta; //Later it is going to be overwritten by the effective viscosity
          out.thermal_expansion_coefficients[q] = 1e-10;
          out.specific_heat[q] = 1;
          out.thermal_conductivities[q] = 1;
          out.compressibilities[q] = 0.0;
          out.entropy_derivative_pressure[q] = 0.0;
          out.entropy_derivative_temperature[q] = 0.0;

          //Create constant value to use for AV
          const double A_o = 1.1e5*exp(-530000/(8.314*in.temperature[q]));
          const double n = 3.5;
          const double Gamma = (A_o/(std::pow(grain_size,0.73)));

          // calculate effective viscosity
          const std::vector<double> &composition = in.composition[q];
          const SymmetricTensor<2,dim> strain_rate = in.strain_rate[q];
          const SymmetricTensor<2,dim> deviatoric_strain_rate
            = (this->get_material_model().is_compressible()
              ?
              strain_rate - 1./3. * trace(strain_rate) * unit_symmetric_tensor<dim>()
              :
              strain_rate);
          SymmetricTensor<2,dim> stress;
          if (anisotropic_viscosity != nullptr)
            {
              std::cout << "Using Aisotropic Stress now!" << std::endl;
              stress = 2 * out.viscosities[q] * anisotropic_viscosity->stress_strain_directors[q] * deviatoric_strain_rate;
            }
          else
            {
              stress = 2 * out.viscosities[q] * deviatoric_strain_rate;
            }


          // std::cout << "q " << acos(q) << std::endl;
          // The computation of the viscosity tensor is only
          // necessary after the simulator has been initialized
          if  ((this->simulator_is_past_initialization()) && (this->get_timestep_number() > 0) && (in.temperature[q]>1000))
            {
              //Get rotation matrix from eigen vectors in compositional fields
              Tensor<2,3> R_CPO;
              R_CPO[0][0] = composition[lpo_bingham_avg_a[0]];
              R_CPO[1][0] = composition[lpo_bingham_avg_a[1]];
              R_CPO[2][0] = composition[lpo_bingham_avg_a[2]];
              R_CPO[0][1] = composition[lpo_bingham_avg_b[0]];
              R_CPO[1][1] = composition[lpo_bingham_avg_b[1]];
              R_CPO[2][1] = composition[lpo_bingham_avg_b[2]];
              R_CPO[0][2] = composition[lpo_bingham_avg_c[0]];
              R_CPO[1][2] = composition[lpo_bingham_avg_c[1]];
              R_CPO[2][2] = composition[lpo_bingham_avg_c[2]];
              // std::cout << "R_CPO " << R_CPO << std::endl;

              //Get eigen values from compositional fields
              const double eigvalue_a1 = composition[lpo_bingham_avg_a[3]];
              const double eigvalue_b1 = composition[lpo_bingham_avg_b[3]];
              const double eigvalue_c1 = composition[lpo_bingham_avg_c[3]];
              const double eigvalue_a2 = composition[lpo_bingham_avg_a[4]];
              const double eigvalue_b2 = composition[lpo_bingham_avg_b[4]];
              const double eigvalue_c2 = composition[lpo_bingham_avg_c[4]];
              const double eigvalue_a3 = composition[lpo_bingham_avg_a[5]];
              const double eigvalue_b3 = composition[lpo_bingham_avg_b[5]];
              const double eigvalue_c3 = composition[lpo_bingham_avg_c[5]];

              //Convert rotation matrix to euler angles phi1, theta, phi2
              double theta = acos(fmod(R_CPO[2][2],1));
              double phi1 = atan(R_CPO[2][0]/R_CPO[2][1]);
              double phi2 = -atan(R_CPO[0][2]/R_CPO[1][2]);

              //Compute Hill Parameters FGHLMN from first two largest eigenvalues of a,b,c axis
              double F = eigvalue_a1*CnI_F[0] + eigvalue_a2*CnI_F[1] + eigvalue_a3*CnI_F[2] + eigvalue_b1*CnI_F[3] + eigvalue_b2*CnI_F[4] + eigvalue_b3*CnI_F[5] + eigvalue_c1*CnI_F[6] + eigvalue_c2*CnI_F[7] + eigvalue_c3*CnI_F[8] + CnI_F[9];
              double G = eigvalue_a1*CnI_G[0] + eigvalue_a2*CnI_G[1] + eigvalue_a3*CnI_G[2] + eigvalue_b1*CnI_G[3] + eigvalue_b2*CnI_G[4] + eigvalue_b3*CnI_G[5] + eigvalue_c1*CnI_G[6] + eigvalue_c2*CnI_G[7] + eigvalue_c3*CnI_G[8] + CnI_G[9];
              double H = eigvalue_a1*CnI_H[0] + eigvalue_a2*CnI_H[1] + eigvalue_a3*CnI_H[2] + eigvalue_b1*CnI_H[3] + eigvalue_b2*CnI_H[4] + eigvalue_b3*CnI_H[5] + eigvalue_c1*CnI_H[6] + eigvalue_c2*CnI_H[7] + eigvalue_c3*CnI_H[8] + CnI_H[9];
              double L = eigvalue_a1*CnI_L[0] + eigvalue_a2*CnI_L[1] + eigvalue_a3*CnI_L[2] + eigvalue_b1*CnI_L[3] + eigvalue_b2*CnI_L[4] + eigvalue_b3*CnI_L[5] + eigvalue_c1*CnI_L[6] + eigvalue_c2*CnI_L[7] + eigvalue_c3*CnI_L[8] + CnI_L[9];
              double M = eigvalue_a1*CnI_M[0] + eigvalue_a2*CnI_M[1] + eigvalue_a3*CnI_M[2] + eigvalue_b1*CnI_M[3] + eigvalue_b2*CnI_M[4] + eigvalue_b3*CnI_M[5] + eigvalue_c1*CnI_M[6] + eigvalue_c2*CnI_M[7] + eigvalue_c3*CnI_M[8] + CnI_M[9];
              double N = eigvalue_a1*CnI_N[0] + eigvalue_a2*CnI_N[1] + eigvalue_a3*CnI_N[2] + eigvalue_b1*CnI_N[3] + eigvalue_b2*CnI_N[4] + eigvalue_b3*CnI_N[5] + eigvalue_c1*CnI_N[6] + eigvalue_c2*CnI_N[7] + eigvalue_c3*CnI_N[8] + CnI_N[9];
              
              //Calculate the rotation matrix from the euler angles
              Tensor<2,3> R = AV<dim>::euler_angles_to_rotation_matrix(phi1, theta, phi2);
              // std::cout << "phi1 " << phi1 << std::endl;
              // std::cout << "theta " << theta << std::endl;
              // std::cout << "phi2 " << phi2 << std::endl;
              // std::cout << "R " << R << std::endl;
              // if (R != R_CPO)
              //   {
              //     std::cout << "R is not the same as R_CPO, R_CPO: " << R_CPO << std::endl;
              //     std::cout << "R: " << R << std::endl;
              //   }

              //Build Rotation matrix
              Tensor<2,6> R_CPO_K;
              R_CPO_K[0][0] = std::pow(R[0][0],2);
              R_CPO_K[0][1] = std::pow(R[0][1],2);
              R_CPO_K[0][2] = std::pow(R[0][2],2);
              R_CPO_K[0][3] = sqrt(2)*R[0][1]*R[0][2];
              R_CPO_K[0][4] = sqrt(2)*R[0][0]*R[0][2];
              R_CPO_K[0][5] = sqrt(2)*R[0][0]*R[0][1];

              R_CPO_K[1][0] = std::pow(R[1][0],2);
              R_CPO_K[1][1] = std::pow(R[1][1],2);
              R_CPO_K[1][2] = std::pow(R[1][2],2);
              R_CPO_K[1][3] = sqrt(2)*R[1][1]*R[1][2];
              R_CPO_K[1][4] = sqrt(2)*R[1][0]*R[1][2];
              R_CPO_K[1][5] = sqrt(2)*R[1][0]*R[1][1];

              R_CPO_K[2][0] = std::pow(R[2][0],2);
              R_CPO_K[2][1] = std::pow(R[2][1],2);
              R_CPO_K[2][2] = std::pow(R[2][2],2);
              R_CPO_K[2][3] = sqrt(2)*R[2][1]*R[2][2];
              R_CPO_K[2][4] = sqrt(2)*R[2][0]*R[2][2];
              R_CPO_K[2][5] = sqrt(2)*R[2][0]*R[2][1];

              R_CPO_K[3][0] = sqrt(2)*R[1][0]*R[2][0];
              R_CPO_K[3][1] = sqrt(2)*R[1][1]*R[2][1];
              R_CPO_K[3][2] = sqrt(2)*R[1][2]*R[2][2];
              R_CPO_K[3][3] = R[1][1]*R[2][2]+R[1][2]*R[2][1];
              R_CPO_K[3][4] = R[1][0]*R[2][2]+R[1][2]*R[2][0];
              R_CPO_K[3][5] = R[1][0]*R[2][1]+R[1][1]*R[2][0];

              R_CPO_K[4][0] = sqrt(2)*R[0][0]*R[2][0];
              R_CPO_K[4][1] = sqrt(2)*R[0][1]*R[2][1];
              R_CPO_K[4][2] = sqrt(2)*R[0][2]*R[2][2];
              R_CPO_K[4][3] = R[0][1]*R[2][2]+R[0][2]*R[2][1];
              R_CPO_K[4][4] = R[0][0]*R[2][2]+R[0][2]*R[2][0];
              R_CPO_K[4][5] = R[0][0]*R[2][1]+R[0][1]*R[2][0];

              R_CPO_K[5][0] = sqrt(2)*R[0][0]*R[1][0];
              R_CPO_K[5][1] = sqrt(2)*R[0][1]*R[1][1];
              R_CPO_K[5][2] = sqrt(2)*R[0][2]*R[1][2];
              R_CPO_K[5][3] = R[0][1]*R[1][2]+R[0][2]*R[1][1];
              R_CPO_K[5][4] = R[0][0]*R[1][2]+R[0][2]*R[2][0];
              R_CPO_K[5][5] = R[0][0]*R[1][1]+R[0][1]*R[1][0];

              //Calculate the fluidity tensor in the LPO frame
              Tensor<2,3> S_CPO=transpose(R)*stress*R;
              // std::cout << "eigvb1 " << R[0][1] << std::endl;

              double Jhill = F*pow((S_CPO[0][0]-S_CPO[1][1]),2)+G*pow((S_CPO[1][1]-S_CPO[2][2]),2)+H*pow((S_CPO[2][2]-S_CPO[0][0]),2)+2*L*pow(S_CPO[1][2],2)+2*M*pow(S_CPO[0][2],2)+2*N*pow(S_CPO[0][1],2);
              // std::cout << "SCPO0 " << S_CPO[0][0] << std::endl;
              // std::cout << "SCP11 " << S_CPO[1][1] << std::endl;
              // std::cout << "SCP22 " << S_CPO[2][2] << std::endl;
              // std::cout << "Gamma " << Gamma << std::endl;
              // std::cout << "Jhill " << Jhill << std::endl;

              SymmetricTensor<2,6> invA;
              invA[0][0] = (F+H)/(F*H+F*G+G*H);
              invA[0][1] = (-2*G-F)/(F*H+F*G+G*H);
              invA[0][2] = (-2*G-F)/(F*H+F*G+G*H);
              invA[1][1] = (G+H)/(F*H+F*G+G*H);
              invA[1][2] = G/(F*H+F*G+G*H);
              invA[2][2] = (F+G)/(F*H+F*G+G*H);
              invA[3][3] = 2/L;
              invA[4][4] = 2/M;
              invA[5][5] = 2/N;

              Tensor<2,6> V = R_CPO_K * invA * transpose(R_CPO_K);
              // std::cout << "FT1 " << FluidityTensor[0][0] << std::endl;

              // Overwrite the scalar viscosity with an effective viscosity
              // double stress_eq = std::sqrt(3.0*AV<dim>::J2_second_invariant(stress, min_strain_rate));
              // double strain_rate_eq = std::sqrt((4./3.)*AV<dim>::J2_second_invariant(strain_rate, min_strain_rate));
              out.viscosities[q] = (1 / (Gamma * std::pow(Jhill,(n-1)/2)));
              // std::cout << "stress_eq " << stress_eq << std::endl;
              // std::cout << "strainrate_eq " << strain_rate_eq << std::endl;
              // std::cout << "Effective viscosity " << std::abs(stress_eq/strain_rate_eq) << std::endl;

              AssertThrow(out.viscosities[q] != 0,
                          ExcMessage("Viscosity should not be 0"));
              AssertThrow(isfinite(out.viscosities[q]),
                          ExcMessage("Viscosity should not be finite"));

              // Compute Viscosity tensor as inverse of the fluidity tensor
              SymmetricTensor<4,dim> V_r4, ViscoTensor_r4;
              V_r4[0][0][0][0]=V[0][0];
              V_r4[0][0][1][1]=V[0][1];
              V_r4[1][1][0][0]=V[0][1];
              V_r4[0][0][2][2]=V[0][2];
              V_r4[2][2][0][0]=V[0][2];
              V_r4[0][0][1][2]=V[0][3];
              V_r4[1][2][0][0]=V[0][3];
              V_r4[0][0][0][2]=V[0][4];
              V_r4[0][2][0][0]=V[0][4];
              V_r4[0][0][0][1]=V[0][5];
              V_r4[0][1][0][0]=V[0][5];
              V_r4[1][1][1][1]=V[1][1];
              V_r4[1][1][2][2]=V[1][2];
              V_r4[2][2][1][1]=V[1][2];
              V_r4[1][1][1][2]=V[1][3];
              V_r4[1][2][1][1]=V[1][3];
              V_r4[1][1][0][2]=V[1][4];
              V_r4[0][2][1][1]=V[1][4];
              V_r4[1][1][0][1]=V[1][5];
              V_r4[0][1][1][1]=V[1][5];
              V_r4[2][2][2][2]=V[2][2];
              V_r4[2][2][1][2]=V[2][3];
              V_r4[1][2][2][2]=V[2][3];
              V_r4[2][2][0][2]=V[2][4];
              V_r4[0][2][2][2]=V[2][4];
              V_r4[2][2][0][1]=V[2][5];
              V_r4[0][1][2][2]=V[2][5];
              V_r4[1][2][1][2]=V[3][3]/2.; //Need to change back later
              V_r4[1][2][0][2]=V[3][4];
              V_r4[0][2][1][2]=V[3][4];
              V_r4[1][2][0][1]=V[3][5];
              V_r4[0][1][1][2]=V[3][5];
              V_r4[0][2][0][1]=V[4][4]/2.;//Need to change back later
              V_r4[0][2][0][1]=V[4][5];
              V_r4[0][1][0][2]=V[4][5];
              V_r4[0][1][0][1]=V[5][5]/2.;//Need to change back later

              for (int i = 0; i < dim; i++)
                {
                  for (int j = 0; j < dim; j++)
                    {

                      for (int k = 0; k<dim; k++)
                        {
                          for (int l = 0; l<dim; l++)
                            {
                              //std::cout<<"V"<<i+1<<j+1<<k+1<<l+1<<"is: "<<V[i][j][k][l]<<std::endl;
                              ViscoTensor_r4[i][j][k][l]= V_r4[i][j][k][l];///std::pow(AV<dim>::J2_second_invariant(Stress, min_strain_rate),(1.25));
                            }
                        }
                    }
                }

              if (anisotropic_viscosity != nullptr)
                {
                  anisotropic_viscosity->stress_strain_directors[q] = ViscoTensor_r4;

                }

            }

        }
    }




    template <int dim>
    bool
    LPO_AV_3D<dim>::is_compressible () const
    {
      return false;
    }



    template <int dim>
    double
    LPO_AV_3D<dim>::reference_viscosity () const
    {
      return 1e20;
    }



    template <int dim>
    void
    LPO_AV_3D<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("AV Hill");
        {

          equation_of_state.parse_parameters (prm);
          eta = prm.get_double("Reference viscosity");
          min_strain_rate = prm.get_double("Minimum strain rate");
          grain_size = prm.get_double("Grain size");
          CnI_F = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for F")));
          CnI_G = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for G")));
          CnI_H = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for H")));
          CnI_L = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for L")));
          CnI_M = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for M")));
          CnI_N = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Coefficients and intercept for N")));

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    LPO_AV_3D<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("AV Hill");
        {
          EquationOfState::LinearizedIncompressible<dim>::declare_parameters (prm);
          prm.declare_entry ("Coefficients and intercept for F", "-0.4602, -3.5366, 0, -0.4521, -2.3999, 0, 1.9207, 0.4799, 0, 1.9214",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter F.");
          prm.declare_entry ("Coefficients and intercept for G", "-1.7559, -0.0666, 0, -0.0677, -0.8096, 0, 1.0592, 0.6844, 0, 0.7957",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter G.");
          prm.declare_entry ("Coefficients and intercept for H", "-0.4608, -2.9948, 0, 0.5591, -1.9798, 0, 1.1741, 0.8055, 0, 1.4096",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter H.");
          prm.declare_entry ("Coefficients and intercept for L", "-1.2098, -0.1456, 0, -0.1280, 0.5754, 0, -0.6839, -0.2313, 0, 2.0596",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter L.");
          prm.declare_entry ("Coefficients and intercept for M", "0.7074, -1.0832, 0, 1.7936, -0.8183, 0, -1.8465, -0.1618, 0, 1.8901",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter M.");
          prm.declare_entry ("Coefficients and intercept for N", "0.9814, 0.1760, 0, -1.9047, 0.0574, 0, 1.5909, 0.0667, 0, 1.1061",
                             Patterns::List(Patterns::Double()),
                             "6 Coefficients and 1 intercept to compute the Hill Parameter N.");
          prm.declare_entry ("Reference viscosity", "1e20",
                             Patterns::Double(),
                             "Magnitude of reference viscosity.");
          prm.declare_entry ("Minimum strain rate", "1.4e-20", Patterns::Double(),
                             "Stabilizes strain dependent viscosity. Units: \\si{\\per\\second}");
          prm.declare_entry ("Grain size", "1000",
                             Patterns::Double(),
                             "Olivine anisotropic viscosity is dependent of grain size. Value is given in microns");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    LPO_AV_3D<dim>::create_additional_named_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (out.template get_additional_output<AV<dim>>() == nullptr)
        {
          const unsigned int n_points = out.n_evaluation_points();
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::AV<dim>> (n_points));
        }
    }
  }
}



// explicit instantiations
namespace aspect
{
  namespace Assemblers
  {
#define INSTANTIATE(dim) \
  template class StokesPreconditionerAV<dim>; \
  template class StokesIncompressibleTermsAV<dim>; \
  //template class StokesBoundaryTractionAV<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)
  }

  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(ShearHeatingAV,
                                  "LPO_AV_3D anisotropic shear heating",
                                  "Implementation of a standard model for shear heating. "
                                  "Adds the term: "
                                  "$  2 \\eta \\left( \\varepsilon - \\frac{1}{3} \\text{tr} "
                                  "\\varepsilon \\mathbf 1 \\right) : \\left( \\varepsilon - \\frac{1}{3} "
                                  "\\text{tr} \\varepsilon \\mathbf 1 \\right)$ to the "
                                  "right-hand side of the temperature equation.")
  }



  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(LPO_AV_3D,
                                   "LPO Anisotropic Viscosity Hill material",
                                   "Olivine LPO related viscous anisotropy based on the Simple material model")
  }
}
