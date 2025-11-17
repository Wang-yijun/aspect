/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/material_model/additional_outputs/anisotropic_viscosity.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      template <int dim>
      std::vector<std::string> make_anisotropic_viscosity_additional_outputs_names()
      {
        std::vector<std::string> names;

        for (unsigned int i = 0; i < Tensor<4,dim>::n_independent_components ; ++i)
          {
            TableIndices<4> indices(Tensor<4,dim>::unrolled_to_component_indices(i));
            names.push_back("anisotropic_viscosity"+std::to_string(indices[0])+std::to_string(indices[1])+std::to_string(indices[2])+std::to_string(indices[3]));
          }
        return names;
      }
    }



    template <int dim>
    AnisotropicViscosity<dim>::AnisotropicViscosity (const unsigned int n_points)
      :
      NamedAdditionalMaterialOutputs<dim>(make_anisotropic_viscosity_additional_outputs_names<dim>()),
      stress_strain_directors(n_points, dealii::identity_tensor<dim> ())
    {}



    template <int dim>
    std::vector<double>
    AnisotropicViscosity<dim>::get_nth_output(const unsigned int idx) const
    {
      std::vector<double> output(stress_strain_directors.size());
      for (unsigned int i = 0; i < stress_strain_directors.size() ; ++i)
        {
          output[i]= stress_strain_directors[i][Tensor<4,dim>::unrolled_to_component_indices(idx)];
        }
      return output;
    }

    template<int dim>
    Tensor<2,3>
    AnisotropicViscosity<dim>::euler_angles_to_rotation_matrix(double phi1, double theta, double phi2)
    {
      Tensor<2,3> rot_matrix;
      //R3*R2*R1 ZXZ rotation. Note it is not exactly the same as in utilities.cc
      rot_matrix[0][0] = cos(phi2)*cos(phi1) - cos(theta)*sin(phi1)*sin(phi2); //
      rot_matrix[0][1] = -cos(phi2)*sin(phi1) - cos(theta)*cos(phi1)*sin(phi2); //cos(phi2)*sin(phi1) + cos(theta)*cos(phi1)*sin(phi2);
      rot_matrix[0][2] = sin(phi2)*sin(theta);
      rot_matrix[1][0] = sin(phi2)*cos(phi1) + cos(theta)*sin(phi1)*cos(phi2); //-sin(phi2)*cos(phi1) - cos(theta)*sin(phi1)*cos(phi2);
      rot_matrix[1][1] = -sin(phi2)*sin(phi1) + cos(theta)*cos(phi1)*cos(phi2);
      rot_matrix[1][2] = -cos(phi2)*sin(theta); //cos(phi2)*sin(theta);
      rot_matrix[2][0] = sin(theta)*sin(phi1);
      rot_matrix[2][1] = sin(theta)*cos(phi1); //-sin(theta)*cos(phi1);
      rot_matrix[2][2] = cos(theta); //
      AssertThrow(rot_matrix[2][2] <= 1.0, ExcMessage("rot_matrix[2][2] > 1.0"));
      return rot_matrix;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  template class AnisotropicViscosity<dim>;

    ASPECT_INSTANTIATE(INSTANTIATE)

#undef INSTANTIATE
  }
}
