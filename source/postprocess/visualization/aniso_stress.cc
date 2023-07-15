/*
  Copyright (C) 2011 - 2021 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/aniso_stress.h>
#include <aspect/material_model/AV_3D_simpler.h>


namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      AnisoStress<dim>::
      AnisoStress ()
        :
        DataPostprocessorTensor<dim> ("aniso_stress",
                                      update_values | update_gradients | update_quadrature_points)
      {}
    


      template <int dim>
      void
      AnisoStress<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double> > &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points,    ExcInternalError());
        Assert ((computed_quantities[0].size() == Tensor<2,dim>::n_independent_components),
                ExcInternalError());
        Assert (input_data.solution_values[0].size() == this->introspection().n_components,   ExcInternalError());
        Assert (input_data.solution_gradients[0].size() == this->introspection().n_components,  ExcInternalError());

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        this->get_material_model().create_additional_named_outputs(out);
        const MaterialModel::AnisotropicViscosity<dim> *anisotropic_viscosity =
          out.template get_additional_output<MaterialModel::AnisotropicViscosity<dim> >();

        this->get_material_model().evaluate(in, out);

        // ...and use it to compute the stresses
        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            const SymmetricTensor<2,dim> strain_rate = in.strain_rate[q];
            const SymmetricTensor<2,dim> deviatoric_strain_rate
              = (this->get_material_model().is_compressible()
                 ?
                 strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                 :
                 strain_rate);

            const double eta = out.viscosities[q];

            SymmetricTensor<2,dim> aniso_stress;
            if (anisotropic_viscosity != nullptr) // when this statement is not used, model runs into segmentation fault. With this though, it's always the second term that is evaluated
              {
                aniso_stress= -2.*eta*anisotropic_viscosity->stress_strain_directors[q]*deviatoric_strain_rate;
                //std::cout << "Anisotropic stress: " << aniso_stress << std::endl;
              }
            else
              {
                aniso_stress= -2.*eta*deviatoric_strain_rate;
              }

            for (unsigned int d=0; d<dim; ++d)
              for (unsigned int e=0; e<dim; ++e)
                computed_quantities[q][Tensor<2,dim>::component_to_unrolled_index(TableIndices<2>(d,e))]
                  = aniso_stress[d][e];
          }
      
        // average the values if requested
        const auto &viz = this->get_postprocess_manager().template get_matching_postprocessor<Postprocess::Visualization<dim> >();
        if (!viz.output_pointwise_stress_and_strain())
          average_quantities(computed_quantities);
      }

      template <int dim>
      void
      AnisoStress<dim>::
      create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        this->get_material_model().create_additional_named_outputs(out);
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(AnisoStress,
                                                  "Aniso stress",
                                                  "A visualization output object that generates output "
                                                  "for the 6 (in 3d) components of the shear stress "
                                                  "tensor, i.e., for the components of the tensor "
                                                  "$-2\\eta\\varepsilon(\\mathbf u)$ "
                                                  "in the incompressible case and "
                                                  "$-2\\eta\\left[\\varepsilon(\\mathbf u)-"
                                                  "\\tfrac 13(\\textrm{tr}\\;\\varepsilon(\\mathbf u))\\mathbf I\\right]$ "
                                                  "in the compressible case. If elasticity is used, the "
                                                  "elastic contribution is being accounted for. The shear "
                                                  "stress differs from the full stress tensor "
                                                  "by the absence of the pressure. Note that the convention "
                                                  "of positive compressive stress is followed. ")
    }
  }
}
