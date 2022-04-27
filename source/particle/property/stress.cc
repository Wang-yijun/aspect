/*
  Copyright (C) 2015 - 2021 by the authors of the ASPECT code.

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

#include <aspect/particle/property/stress.h>
#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
    namespace Particle
    {
        namespace Property
        {
        template <int dim>
        void
        Stress<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                      std::vector<double> &data) const
        {
        const static Tensor<2,dim> identity = unit_symmetric_tensor<dim>();
        for (unsigned int i = 0; i < Tensor<2,dim>::n_independent_components ; ++i) 
            data.push_back(identity[Tensor<2,dim>::unrolled_to_component_indices(i)]);
            
        }

        template <int dim>
        void
        Stress<dim>::update_particle_property(const unsigned int data_position,
                                              //const Point<dim> &position,
                                              const Vector<double> &solution,
                                              const std::vector<Tensor<1,dim> > &gradients,
                                              typename ParticleHandler<dim>::particle_iterator &particle) const
        {
            // Get initial particle properties to be stored in &data
            auto &data = particle->get_properties();

            // Velocity gradients
            Tensor<2,dim> grad_u;
            for (unsigned int d=0; d<dim; ++d)
            grad_u[d] = gradients[d];
            // Calculate strain rate from velocity gradients
            const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
            const SymmetricTensor<2,dim> deviatoric_strain_rate
                = (this->get_material_model().is_compressible()
                    ?
                    strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                    :
                    strain_rate);
            //std::cout<<"strainrate: "<<strain_rate<<std::endl;	

            // Get pressure and temperature from solution
            const double pressure = solution[this->introspection().component_indices.pressure];
            const double temperature = solution[this->introspection().component_indices.temperature;
            // Get velocity
            Tensor<1,dim> velocity;
            for (unsigned int i = 0; i < dim; ++i)
            velocity[i] = solution[this->introspection().component_indices.velocities[i]];
            // Get the composition of the particle
            std::vector<double> compositions;
            for (unsigned int i = 0; i < this->n_compositional_fields(); i++)
                {
                    const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
                    compositions.push_back(solution[solution_component]);
                }

            MaterialModel::MaterialModelInputs<dim> in(1, this->n_compositional_fields());
            //in.position[0] = position;
            in.temperature[0] = temperature;
            in.pressure[0] = pressure;
            in.velocity[0] = velocity;
            in.composition[0] = compositions;
            in.strain_rate[0] = strain_rate;
            MaterialModel::MaterialModelOutputs<dim> out(1, this->n_compositional_fields());
            this->get_material_model().evaluate(in, out);
            
            const double eta = out.viscosities[0];
            
            // Calculate stress from viscosity and strain rate
            const SymmetricTensor<2,dim> stress = 2 * eta * deviatoric_strain_rate + 
                                                  pressure * unit_symmetric_tensor<dim>();
            
            // Write particle properties (stress tensor)
            for (unsigned int i = 0; i < Tensor<2,dim>::n_independent_components ; ++i) {            
                data[data_position + i] = stress[Tensor<2,dim>::unrolled_to_component_indices(i)];
            }
            
            
        
        }

        template <int dim>
        UpdateTimeFlags
        Stress<dim>::need_update() const
        {
            return update_time_step;
        }

        template <int dim>
        UpdateFlags
        Stress<dim>::get_needed_update_flags() const
        {
            return update_gradients;
        }

        template <int dim>
        std::vector<std::pair<std::string, unsigned int> >
        Stress<dim>::get_property_information() const
        {
            const unsigned int n_components = Tensor<2,dim>::n_independent_components;
            const std::vector<std::pair<std::string,unsigned int> > property_information (1,std::make_pair("strainrate",n_components));
            return property_information;
        }
        }
    }
}

// explicit instantiations
namespace aspect
{
    namespace Particle
    {
        namespace Property
        {
        ASPECT_REGISTER_PARTICLE_PROPERTY(Stress,
                                            "stress",
                                            "Implementation of a plugin in which the particle "
                                            "property is defined as the recent stress at "
                                            "this position.")
        }
    }
}
