/*
 Copyright (C) 2023 by the authors of the ASPECT code.

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

#ifndef _aspect_particle_property_cpo_bingham_average_h
#define _aspect_particle_property_cpo_bingham_average_h

#include <aspect/particle/property/interface.h>
#include <aspect/particle/property/crystal_preferred_orientation.h>
#include <aspect/simulator_access.h>

#include <array>
#include <random>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      /**
       * Computes the Bingham average of the CPO particle properties.
       * See https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html for more info.
       *
       * The layout of the data vector per particle is the following (note that for this plugin the following dim's are always 3):
       * 1 phi1 of olivine                  -> 1 (dim) doubles, starts at:
       *                                         data_position + 1,
       * 2 eigenvalues of a axis of olivine -> 3 (dim) doubles, starts at:
       *                                         data_position + 2,
       * 3 theta of olivine                 -> 1 (dim) doubles, starts at:
       *                                         data_position + 5,
       * 4 eigenvalues of b axis of olivine -> 3 (dim) doubles, starts at:
       *                                         data_position + 6,
       * 5 phi2 of olivine                  -> 1 (dim) doubles, starts at:
       *                                         data_position + 9,
       * 6 eigenvalues of c axis of olivine -> 3 (dim) doubles, starts at:
       *                                         data_position + 10,
       * 7 phi1 of enstatite                   -> 1 (dim) doubles, starts at:
       *                                          data_position + 13,
       * 8 eigenvalues of a axis of enstatite  -> 3 (dim) doubles, starts at:
       *                                          data_position + 14,
       * 9 theta of enstatite                  -> 1 (dim) doubles, starts at:
       *                                          data_position + 17,
       * 10 eigenvalues of a axis of enstatite -> 3 (dim) doubles, starts at:
       *                                          data_position + 18,
       * 11 phi2 of enstatite                  -> 1 (dim) doubles, starts at:
       *                                          data_position + 21,
       * 12 eigenvalues of a axis of enstatite -> 3 (dim) doubles, starts at:
       *                                          data_position + 22,
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class CpoBinghamAverageEA : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * constructor
           */
          CpoBinghamAverageEA() = default;

          /**
           * Initialization function. This function is called once at the
           * beginning of the program after parse_parameters is run.
           */
          void
          initialize () override;

          /**
           * Initialization function. This function is called once at the
           * creation of every particle for every property to initialize its
           * value.
           *
           * @param [in] position The current particle position.
           * @param [in,out] particle_properties The properties of the particle
           * that is initialized within the call of this function. The purpose
           * of this function should be to extend this vector by a number of
           * properties.
           */
          void
          initialize_one_particle_property (const Point<dim> &position,
                                            std::vector<double> &particle_properties) const override;

          /**
           * Update function. This function is called every time an update is
           * request by need_update() for every particle for every property.
           * It is obvious that
           * this function is called a lot, so its code should be efficient.
           * The interface provides a default implementation that does nothing,
           * therefore derived plugins that do not require an update do not
           * need to implement this function.
           *
           * @param [in] data_position An unsigned integer that denotes which
           * component of the particle property vector is associated with the
           * current property. For properties that own several components it
           * denotes the first component of this property, all other components
           * fill consecutive entries in the @p particle_properties vector.
           *
           * @param [in] solution The values of the solution variables at the
           * current particle position.
           *
           * @param [in] gradients The gradients of the solution variables at
           * the current particle position.
           *
           * @param [in,out] particle The particle that is updated within
           * the call of this function. The particle location can be accessed
           * using particle->get_location() and its properties using
           * particle->get_properties().
           */
          void
          update_particle_property (const unsigned int data_position,
                                    const Vector<double> &solution,
                                    const std::vector<Tensor<1,dim>> &gradients,
                                    typename ParticleHandler<dim>::particle_iterator &particle) const override;

          /**
           * This implementation tells the particle manager that
           * we need to update particle properties every time step.
           */
          UpdateTimeFlags
          need_update () const override;

          /**
           * Return which data has to be provided to update the property.
           * The integrated strains needs the gradients of the velocity.
           */
          UpdateFlags
          get_needed_update_flags () const override;

          /**
           * Set up the information about the names and number of components
           * this property requires.
           *
           * @return A vector that contains pairs of the property names and the
           * number of components this property plugin defines.
           */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

          /**
           * Computes the Bingham average. This is a way to average directions. The method comes from
           * https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html, where it is explained
           * with the anology that each vector/pole has a weight on a sphere. This method allows to find
           * the moment of inertia for spinning that sphere. Here we just use it to get three averaged
           * axis associated with the densest clustering of points for each axis. the a to c axis vectors
           * are stored in the first to last array respectively.
           */
          std::array<std::array<double,4>,3>
          compute_bingham_average(std::vector<Tensor<2,3>> matrices) const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm) override;

        private:
          /**
           * stores the position of the cpo data in the particle property vector
           */
          unsigned int cpo_data_position;

          /**
           * A pointer to the crystal preferred orientation particle property.
           * Avoids repeated searches for that property.
           */
          std::unique_ptr<const Particle::Property::CrystalPreferredOrientation<dim>> cpo_particle_property;

          /**
           * Random number generator. For reproducibility of tests it is
           * initialized in the constructor with a constant plus the MPI rank.
           */
          mutable std::mt19937 random_number_generator;

          /**
           * the random number generator seed used to initialize the random number generator.
           */
          unsigned int random_number_seed;

          /**
           * Number of grains
           */
          unsigned int n_grains;

          /**
           * Number of minerals
           */
          unsigned int n_minerals;

          /**
           * when doing the random draw volume weighting, this sets how many samples are taken.
           */
          unsigned int n_samples;

      };
    }
  }
}

#endif