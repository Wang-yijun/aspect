/*
  Copyright (C) 2015 - 2017 by the authors of the ASPECT code.

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

//#include <cstdlib>
#include <aspect/particle/property/lpo_AV_Ss_tensor.h>
#include <aspect/particle/property/lpo.h>
#include <aspect/particle/world.h>

#include <aspect/utilities.h>

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {


      template <int dim>
      LpoSsTensor<dim>::LpoSsTensor ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;

        

        // tensors of indices
        indices_tensor[0][0] = 0;
        indices_tensor[0][1] = 5;
        indices_tensor[0][2] = 4;
        indices_tensor[1][0] = 5;
        indices_tensor[1][1] = 1;
        indices_tensor[1][2] = 3;
        indices_tensor[2][0] = 4;
        indices_tensor[2][1] = 3;
        indices_tensor[2][2] = 2;

        // vectors of indices
        indices_vector_1.resize(6);
        indices_vector_1[0] = 0;
        indices_vector_1[1] = 1;
        indices_vector_1[2] = 2;
        indices_vector_1[3] = 1;
        indices_vector_1[4] = 2;
        indices_vector_1[5] = 0;

        indices_vector_2.resize(6);
        indices_vector_2[0] = 0;
        indices_vector_2[1] = 1;
        indices_vector_2[2] = 2;
        indices_vector_2[3] = 2;
        indices_vector_2[4] = 0;
        indices_vector_2[5] = 1;
      }

      template <int dim>
      void
      LpoSsTensor<dim>::initialize ()
      {
        // todo: check wheter this works correctly. Since the get_random_number function takes a reference
        // to the random_number_generator function, changing the function should mean that I have to update the
        // get_random_number function as well. But I will need to test this.
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);

        const auto &manager = this->get_particle_world().get_property_manager();
        AssertThrow(manager.plugin_name_exists("lpo"),
                    ExcMessage("No lpo property plugin found."));
        Assert(manager.plugin_name_exists("lpo Ss tensor"),
               ExcMessage("No LPO aniso stress property plugin found."));

        AssertThrow(manager.check_plugin_order("lpo","lpo Ss tensor"),
                    ExcMessage("To use the lpo Ss tensor plugin, the lpo plugin need to be defined before this plugin."));

        lpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo"));

        if(dim == 2)
        {
          Assert(false,ExcMessage("This PROPERTY is not implemented for 2D."));
          
        }


      }


      template <int dim>
      SymmetricTensor<2,dim>
      LpoSsTensor<dim>::compute_S_tensor (const SymmetricTensor<2,dim> &strain_rate,
                                          const double grain_size,
                                          const std::vector<std::vector<Tensor<2,3> > > &a_cosine_matrices_grains,
                                          const std::vector<unsigned int> &deformation_type,
                                          const double &temperature) const
      {
        Assert(false,ExcMessage("This PROPERTY is not implemented for 2D."));
      }
      /*template <>
      SymmetricTensor<2,2>
      LpoSsTensor<2>::compute_S_tensor (const SymmetricTensor<2,2> &strain_rate,
                                          const double &grain_size,
                                          const std::vector<std::vector<Tensor<2,3> > > &a_cosine_matrices_grains,
                                          const std::vector<unsigned int> &deformation_type,
                                          const double &temperature) const;
      
      {
          Assert(false,ExcMessage("This PROPERTY is not implemented for 2D."));  
      }*/
      template <>
      SymmetricTensor<2,3>
      LpoSsTensor<3>::compute_S_tensor (const SymmetricTensor<2,3> &strain_rate,
                                          const double grain_size,
                                          const std::vector<std::vector<Tensor<2,3> > > &a_cosine_matrices_grains,
                                          const std::vector<unsigned int> &deformation_type,
                                          const double &temperature) const
      
      {
        const size_t n_minerals_local = a_cosine_matrices_grains.size();
        std::cout<<"n_minerals_local: "<< n_minerals_local<<  std::endl;
        const size_t n_grains_local = a_cosine_matrices_grains[0].size();
        std::cout<<"n_grains_local: "<< n_grains_local<<  std::endl;
        SymmetricTensor<2,3, double> S_sum;
        const int dim=3;
        double nFo = 4.1;
        double A0 = 1.1e5*std::exp(-530000/8.314/temperature);
        std::cout<<"T: "<<temperature<<std::endl;
        std::cout<<"A0: "<<A0<<std::endl;
        FullMatrix<double> Schm(6,3); //Schmid tensor, 6x3 matrix
        FullMatrix<double> pinvschm(3,6); //pseudoinverse of Schmid tensor, 3x6 matrix
        
          
        Schm[3][2] = 1;
        Schm[4][1] = 1;
        Schm[5][0] = 1;
        pinvschm[0][5] = 1;
        pinvschm[1][4] = 1;
        pinvschm[2][3] = 1;

        for (size_t mineral_i = 0; mineral_i < n_minerals_local; mineral_i++)
          {
            //std::cout<<"Def style: "<<deformation_type[mineral_i]<<std::endl;
            Tensor<1,3> A_ss; //A_ss is the invers of the minimum resolved stress on the slip systems on the nth power
            //std::cout<<"A_ss: "<<A_ss<<  std::endl; //correct values
            if (deformation_type[mineral_i] == (unsigned int)DeformationTypeSelector::Enstatite)
              {
                A_ss[0] = 1.;
                A_ss[1] = 1.;
                A_ss[2] = 1.;
                //std::cout<<"Setting Enstatite A_ss: "<<A_ss<<std::endl;
              }
            else
              {
                A_ss[0] = 139.2525;
                A_ss[1] = 214.4907;
                A_ss[2] = 0.3520;
                //std::cout<<"Setting Olivine A_ss: "<<A_ss<<std::endl;
              }
            
            std::cout<<"A_ss: "<<A_ss<<std::endl;
            for (size_t i = 0; i < n_grains_local; i++)
              {
                std::cout<<"strain rate: "<<strain_rate<<std::endl;
                Tensor<2,3> R = a_cosine_matrices_grains[mineral_i][i];
                std::cout<<"Rotation matrix: "<<R<<std::endl;
                SymmetricTensor<2,3> Rate_grain=symmetrize(R*strain_rate*transpose(R));
                std::cout<<"Rate_grain "<<Rate_grain<<  std::endl;
                std::array<std::pair<double, Tensor<1, 3>>, 3> Rate_gr_eig = eigenvectors(Rate_grain,SymmetricTensorEigenvectorMethod::jacobi);
                std::cout<<"Rate_grain eigen values: "<<Rate_gr_eig[0].first<<"; "<<Rate_gr_eig[1].first<<"; "<<Rate_gr_eig[2].first<<std::endl;
                double inv2=std::pow(Rate_gr_eig[0].first-Rate_gr_eig[1].first,2)
                          +std::pow(Rate_gr_eig[1].first-Rate_gr_eig[2].first,2)
                          +std::pow(Rate_gr_eig[2].first-Rate_gr_eig[0].first,2);
                std::cout<<"inv2: "<<inv2<<  std::endl;

                FullMatrix<double> Rate_grain_voigt(6,1);
                Rate_grain_voigt[0][0]=Rate_grain[0][0];
                Rate_grain_voigt[1][0]=Rate_grain[1][1];
                Rate_grain_voigt[2][0]=Rate_grain[2][2];
                Rate_grain_voigt[3][0]=2*Rate_grain[1][2];
                Rate_grain_voigt[4][0]=2*Rate_grain[0][2];
                Rate_grain_voigt[5][0]=2*Rate_grain[0][1];
                
                std::cout<<"Rate_grain_voigt: "<<Rate_grain_voigt[0][0]<<"; "<<Rate_grain_voigt[1][0]<<"; "<<Rate_grain_voigt[2][0]<<"; "<<Rate_grain_voigt[3][0]<<"; "<<Rate_grain_voigt[4][0]<<"; "<<Rate_grain_voigt[5][0]<<std::endl;

                FullMatrix<double> r_ss(3,1); //Optimazition to find shear strain rate on slip system
                FullMatrix<double> r_gc_v(6,1); //strain rate tensor for grain in Voigt notation and crystal reference frame
                pinvschm.mmult(r_ss, Rate_grain_voigt);
                std::cout<<"r_ss: "<<r_ss[0][0]<<"; "<<r_ss[1][0]<<"; "<<r_ss[2][0]<<std::endl;
                Schm.mmult(r_gc_v,r_ss);

                SymmetricTensor<2,3> r_gc; 
                r_gc[0][0]=r_gc_v[0][0];
                r_gc[1][1]=r_gc_v[1][0];
                r_gc[2][2]=r_gc_v[2][0];
                r_gc[1][2]=0.5*r_gc_v[3][0];
                r_gc[0][2]=0.5*r_gc_v[4][0];
                r_gc[0][1]=0.5*r_gc_v[5][0];
                std::cout<<"r_gc:  "<<r_gc<<  std::endl;
                std::array<std::pair<double, Tensor<1, dim>>, dim> r_gc_eig = eigenvectors(r_gc, SymmetricTensorEigenvectorMethod::jacobi);
		      
                double inv2best =std::pow(r_gc_eig[0].first-r_gc_eig[1].first,2)
                              +std::pow(r_gc_eig[1].first-r_gc_eig[2].first,2)
                              +std::pow(r_gc_eig[0].first-r_gc_eig[2].first,2);

		          for (unsigned int i=0; i<dim; ++i)
                  {
                    r_ss[i][0]=r_ss[i][0]*std::pow(inv2/inv2best,0);
                  }
                FullMatrix<double> tau_ss(3,1);
                //std::cout<<"std::copysignf(1.0,r_ss[0][0]): "<<std::copysignf(1.0,r_ss[0][0])<<  std::endl;
                //std::cout<<"1.0/A_ss[0]: "<<1.0/A_ss[0]<<  std::endl;
                //std::cout<<"1.0/A0: "<<1.0/A0<<std::endl;
                AssertThrow(isfinite(1./A0),
                    ExcMessage("1/A0 is infinite"))
                //std::cout<<"std::pow(grain_size,0.73): "<<std::pow(grain_size,0.73)<<std::endl;
                //std::cout<<"std::fabs(r_ss[0][0]/2) "<<std::fabs(r_ss[0][0]/2)<<std::endl;
                //std::cout<<"tau_ss(1) "<<std::copysignf(1.0,r_ss[0][0])*std::pow(1.0/A_ss[0]*1.0/A0*std::pow(grain_size,0.73)*std::fabs(r_ss[0][0]/2),1.0/nFo)<<std::endl;
		            tau_ss[0][0]= std::copysignf(1.0,r_ss[0][0])*std::pow(1.0/A_ss[0]*1.0/A0*std::pow(grain_size,0.73)*std::fabs(r_ss[0][0]/2),1.0/nFo);
		            tau_ss[1][0]= std::copysignf(1.0,r_ss[1][0])*std::pow(1.0/A_ss[1]*1.0/A0*std::pow(grain_size,0.73)*std::fabs(r_ss[1][0]/2),1.0/nFo);
		            tau_ss[2][0]= std::copysignf(1.0,r_ss[2][0])*std::pow(1.0/A_ss[2]*1.0/A0*std::pow(grain_size,0.73)*std::fabs(r_ss[2][0]/2),1.0/nFo);
                std::cout<<"Tau_ss(0,0): "<<tau_ss[0][0]<<  std::endl;
                std::cout<<"Tau_ss(1,0): "<<tau_ss[1][0]<<  std::endl;
                std::cout<<"Tau_ss(2,0): "<<tau_ss[2][0]<<  std::endl;

                FullMatrix<double>  S_gc_v(6,1);
                Schm.mmult(S_gc_v,tau_ss); //Voigt notation of the resolved stress on the grain
		SymmetricTensor<2,3> S_gc;
                S_gc[0][0] = S_gc_v[0][0];
                S_gc[1][1] = S_gc_v[1][0];
                S_gc[2][2] = S_gc_v[2][0];
                S_gc[1][2] = S_gc_v[3][0];
                S_gc[0][2] = S_gc_v[4][0];
                S_gc[0][1] = S_gc_v[5][0];
		std::cout<<"S_gc: "<<S_gc<<std::endl;
                SymmetricTensor<2,3> S_g= symmetrize(transpose(R)*S_gc*R); //Here instead of making a multidimensional array what I sum at the end, I create S_g and add it to S_sum
                //SymmetricTensor<2,3> S_sum;
                std::cout<<"S_g: "<<S_g<<std::endl;
                S_sum += S_g;
                std::cout<<"S_sum: "<<S_sum<<std::endl;

              }
            S_sum=S_sum/n_grains_local; //Stress for mineralphase

          }
        S_sum=S_sum/n_minerals_local; //stress for particle
        S_sum *= 1e6;
        std::cout<<"S_sum final: "<<S_sum<<  std::endl;

        return S_sum;
      }



      template <int dim>
      void
      LpoSsTensor<dim>::initialize_one_particle_property(const Point<dim> &,
                                                              std::vector<double> &data) const
      {
        std::vector<unsigned int> deformation_type;
        std::vector<double> volume_fraction_mineral;
        std::vector<std::vector<double>> volume_fractions_grains;
        std::vector<std::vector<Tensor<2,3> > > a_cosine_matrices_grains;

        Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         deformation_type,
                                                         volume_fraction_mineral,
                                                         volume_fractions_grains,
                                                         a_cosine_matrices_grains);


        
        Tensor<2,6> Ss_tensor; //The Ss tensor is a compilation of the stresses needed for the calculation of the viscosity tensor

        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < Tensor<2,6>::n_independent_components ; ++i)

          {
            data.push_back(Ss_tensor[Tensor<2,6>::unrolled_to_component_indices(i)]);
          }


      }

      template <int dim>
      void
      LpoSsTensor<dim>::update_one_particle_property(const unsigned int data_position,
                                                          const Point<dim> &,
                                                          const Vector<double> &solution,
                                                          const std::vector<Tensor<1,dim> > &gradients,
                                                          const ArrayView<double> &data) const
      {
        std::vector<unsigned int> deformation_type;
        std::vector<double> volume_fraction_mineral;
        std::vector<std::vector<double>> volume_fractions_grains;
        std::vector<std::vector<Tensor<2,3> > > a_cosine_matrices_grains;

        Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         deformation_type,
                                                         volume_fraction_mineral,
                                                         volume_fractions_grains,
                                                         a_cosine_matrices_grains);
        
        //std::cout<<"a_cos_matrices -size1: "<< a_cosine_matrices_grains.size()<< "a_cos_matrices -size2: "<< a_cosine_matrices_grains[0].size()<< std::endl;
        //std::cout<<"deformation style -size: "<< deformation_type.size()<<  std::endl;

        const double grain_size=1000.0; //micron --> should be an input?
        Tensor<2,6> Ss_tensor; //Initial value 0, because at initial timestep we don't have strain rate

        double temperature = solution[this->introspection().component_indices.temperature];

        if  (this->get_timestep_number() > 0 && temperature > 1000)
        {
          Tensor<2,dim> velocity_gradient;
          for (unsigned int d=0; d<dim; ++d)
          {
            velocity_gradient[d] = gradients[d]; 
          }
          const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);
          //std::cout<<"strain rate: "<< strain_rate<< std::endl;
          double E_eq;
          SymmetricTensor<2,dim> e1, e2, e3, e4, e5, E;
          E=strain_rate;
          E_eq=(1.0/6.0*(std::pow(double (E[0][0] - E[1][1]),2) + std::pow(double (E[1][1] - E[2][2]),2)+std::pow(double (E[2][2] - E[0][0]),2)))+(std::pow(E[0][1],2)+std::pow(E[1][2],2)+std::pow(E[2][0],2));//J2
          E_eq= std::sqrt((4./3.)*E_eq);// Second invariant of strain-rate
        
          AssertThrow(isfinite(1/E.norm()),
                    ExcMessage("Strain rate should be finite")); 
        
          //We define 5 independent strainrates, of which E is the linear combination
          e1[0][0]=E_eq;
          e1[1][1]=E_eq;
          e1[2][2]=-2*E_eq;
          e2[0][0]=E_eq;
          e2[1][1]=-2*E_eq;
          e2[2][2]=E_eq;
          e3[0][1]=E_eq;
          e3[1][0]=E_eq;
          e4[0][2]=E_eq;
          e4[2][0]=E_eq;
          e5[1][2]=E_eq;
          e5[2][1]=E_eq;

          //We calculate the stress response for each strain rate with the micromechanical model
          // AssertThrow(in.temperature[q] != 0,
          //     ExcMessage("Temperature is 0")); 
          //std::cout<<"e1: "<< e1<<std::endl;
          //std::cout<<"grain size: "<< grain_size<<std::endl;
          //std::cout<<"T: "<< temperature<<std::endl;
          SymmetricTensor<2,dim> stress1, stress2, stress3, stress4, stress5, Stress;
          stress1=compute_S_tensor(e1, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          stress2=compute_S_tensor(e2, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          stress3=compute_S_tensor(e3, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          stress4=compute_S_tensor(e4, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          stress5=compute_S_tensor(e5, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          Stress =compute_S_tensor(E, grain_size, a_cosine_matrices_grains, deformation_type, temperature);
          std::cout<<"Stress SS tensor: "<<Stress<<std::endl;
          std::cout<<"strain rate: "<<E<<std::endl;
          // std::cout<<"Rotation matrix: "<<a_cosine_matrices_grains[0][0]<<std::endl;
          for (unsigned int i = 0; i < SymmetricTensor<2,dim>::n_independent_components ; ++i)
          {
            Ss_tensor[0][i] = Stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];
            Ss_tensor[1][i] = stress1[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];
            Ss_tensor[2][i] = stress2[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];
            Ss_tensor[3][i] = stress3[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];
            Ss_tensor[4][i] = stress4[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];
            Ss_tensor[5][i] = stress5[SymmetricTensor<2,dim>::unrolled_to_component_indices(i)];

          }  
        }
        Particle::Property::LpoSsTensor<dim>::store_particle_data(data_position,
                                                                       data,
                                                                       Ss_tensor);
        std::cout<<"Stress again: "<< Ss_tensor[0]<< std::endl;

      }


      template <int dim>
      void
      LpoSsTensor<dim>::load_particle_data(unsigned int lpo_data_position,
                                                const ArrayView<double> &data,
                                                Tensor<2,6> &Ss_tensor)
      {

        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < Tensor<2,6>::n_independent_components ; ++i)
          {
            Ss_tensor[Tensor<2,6>::unrolled_to_component_indices(i)] = data[lpo_data_position + i];
          }

        //for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
        //elastic_tensor[SymmetricTensor<2,6>::unrolled_to_component_indices(i)] = data[lpo_data_position + i];
      }


      template <int dim>
      void
      LpoSsTensor<dim>::store_particle_data(unsigned int lpo_data_position,
                                                 const ArrayView<double> &data,
                                                 Tensor<2,6> &Ss_tensor)
      {
        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < Tensor<2,6>::n_independent_components ; ++i)
          {
            data[lpo_data_position + i] = Ss_tensor[Tensor<2,6>::unrolled_to_component_indices(i)];
          }

      }


      template<int dim>
      Tensor<4,3>
      LpoSsTensor<dim>::rotate_4th_order_tensor(const Tensor<4,3> &input_tensor, const Tensor<2,3> &rotation_tensor)
      {
        Tensor<4,3> output;

        for (unsigned short int i1 = 0; i1 < 3; i1++)
          {
            for (unsigned short int i2 = 0; i2 < 3; i2++)
              {
                for (unsigned short int i3 = 0; i3 < 3; i3++)
                  {
                    for (unsigned short int i4 = 0; i4 < 3; i4++)
                      {
                        for (unsigned short int j1 = 0; j1 < 3; j1++)
                          {
                            for (unsigned short int j2 = 0; j2 < 3; j2++)
                              {
                                for (unsigned short int j3 = 0; j3 < 3; j3++)
                                  {
                                    for (unsigned short int j4 = 0; j4 < 3; j4++)
                                      {
                                        output[i1][i2][i3][i4] = output[i1][i2][i3][i4] + rotation_tensor[i1][j1]*rotation_tensor[i2][j2]*rotation_tensor[i3][j3]*rotation_tensor[i4][j4]*input_tensor[j1][j2][j3][j4];
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }

        return output;
      }

      template<int dim>
      SymmetricTensor<2,6>
      LpoSsTensor<dim>::rotate_6x6_matrix(const Tensor<2,6> &input_tensor, const Tensor<2,3> &rotation_tensor)
      {
        // we can represent the roation of the 4th order tensor as a rotation in the voigt
        // notation by computing $C'=MCM^{-1}$. Because M is orhtogonal we can replace $M^{-1}$
        // with $M^T$ resutling in $C'=MCM^{T}$ (Carcione, J. M. (2007). Wave Fields in Real Media:
        // Wave Propagation in Anisotropic, Anelastic, Porous and Electromagnetic Media. Netherlands:
        // Elsevier Science. Pages 8-9).
        Tensor<2,6> rotation_matrix;
        // top left block
        rotation_matrix[0][0] = rotation_tensor[0][0] * rotation_tensor[0][0];
        rotation_matrix[1][0] = rotation_tensor[1][0] * rotation_tensor[1][0];
        rotation_matrix[2][0] = rotation_tensor[2][0] * rotation_tensor[2][0];
        rotation_matrix[0][1] = rotation_tensor[0][1] * rotation_tensor[0][1];
        rotation_matrix[1][1] = rotation_tensor[1][1] * rotation_tensor[1][1];
        rotation_matrix[2][1] = rotation_tensor[2][1] * rotation_tensor[2][1];
        rotation_matrix[0][2] = rotation_tensor[0][2] * rotation_tensor[0][2];
        rotation_matrix[1][2] = rotation_tensor[1][2] * rotation_tensor[1][2];
        rotation_matrix[2][2] = rotation_tensor[2][2] * rotation_tensor[2][2];

        // top right block
        rotation_matrix[0][3] = 2.0 * rotation_tensor[0][1] * rotation_tensor[0][2];
        rotation_matrix[1][3] = 2.0 * rotation_tensor[1][1] * rotation_tensor[1][2];
        rotation_matrix[2][3] = 2.0 * rotation_tensor[2][1] * rotation_tensor[2][2];
        rotation_matrix[0][4] = 2.0 * rotation_tensor[0][2] * rotation_tensor[0][0];
        rotation_matrix[1][4] = 2.0 * rotation_tensor[1][2] * rotation_tensor[1][0];
        rotation_matrix[2][4] = 2.0 * rotation_tensor[2][2] * rotation_tensor[2][0];
        rotation_matrix[0][5] = 2.0 * rotation_tensor[0][0] * rotation_tensor[0][1];
        rotation_matrix[1][5] = 2.0 * rotation_tensor[1][0] * rotation_tensor[1][1];
        rotation_matrix[2][5] = 2.0 * rotation_tensor[2][0] * rotation_tensor[2][1];

        // bottom left block
        rotation_matrix[3][0] = rotation_tensor[1][0] * rotation_tensor[2][0];
        rotation_matrix[4][0] = rotation_tensor[2][0] * rotation_tensor[0][0];
        rotation_matrix[5][0] = rotation_tensor[0][0] * rotation_tensor[1][0];
        rotation_matrix[3][1] = rotation_tensor[1][1] * rotation_tensor[2][1];
        rotation_matrix[4][1] = rotation_tensor[2][1] * rotation_tensor[0][1];
        rotation_matrix[5][1] = rotation_tensor[0][1] * rotation_tensor[1][1];
        rotation_matrix[3][2] = rotation_tensor[1][2] * rotation_tensor[2][2];
        rotation_matrix[4][2] = rotation_tensor[2][2] * rotation_tensor[0][2];
        rotation_matrix[5][2] = rotation_tensor[0][2] * rotation_tensor[1][2];

        // bottom right block
        rotation_matrix[3][3] = rotation_tensor[1][1] * rotation_tensor[2][2] + rotation_tensor[1][2] * rotation_tensor[2][1];
        rotation_matrix[4][3] = rotation_tensor[0][1] * rotation_tensor[2][2] + rotation_tensor[0][2] * rotation_tensor[2][1];
        rotation_matrix[5][3] = rotation_tensor[0][1] * rotation_tensor[1][2] + rotation_tensor[0][2] * rotation_tensor[1][1];
        rotation_matrix[3][4] = rotation_tensor[1][0] * rotation_tensor[2][2] + rotation_tensor[1][2] * rotation_tensor[2][0];
        rotation_matrix[4][4] = rotation_tensor[0][2] * rotation_tensor[2][0] + rotation_tensor[0][0] * rotation_tensor[2][2];
        rotation_matrix[5][4] = rotation_tensor[0][2] * rotation_tensor[1][0] + rotation_tensor[0][0] * rotation_tensor[1][2];
        rotation_matrix[3][5] = rotation_tensor[1][1] * rotation_tensor[2][0] + rotation_tensor[1][0] * rotation_tensor[2][1];
        rotation_matrix[4][5] = rotation_tensor[0][0] * rotation_tensor[2][1] + rotation_tensor[0][1] * rotation_tensor[2][0];
        rotation_matrix[5][5] = rotation_tensor[0][0] * rotation_tensor[1][1] + rotation_tensor[0][1] * rotation_tensor[1][0];

        Tensor<2,6> rotation_matrix_tranposed = transpose(rotation_matrix);

        return symmetrize((rotation_matrix*input_tensor)*rotation_matrix_tranposed);
      }



      template<int dim>
      SymmetricTensor<2,6>
      LpoSsTensor<dim>::transform_4th_order_tensor_to_6x6_matrix(const Tensor<4,3> &input_tensor)
      {
        SymmetricTensor<2,6> output;

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][i] = input_tensor[i][i][i][i];
          }

        for (unsigned short int i = 1; i < 3; i++)
          {
            output[0][i] = 0.5*(input_tensor[0][0][i][i] + input_tensor[i][i][0][0]);
            //output[0][i] = output[i][0];
          }
        output[1][2]=0.5*(input_tensor[1][1][2][2]+input_tensor[2][2][1][1]);
        //output[2][1]=output[1][2];

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][3]=0.25*(input_tensor[i][i][1][2]+input_tensor[i][i][2][1]+ input_tensor[1][2][i][i]+input_tensor[2][1][i][i]);
            //output[3][i]=output[i][3];
          }

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][4]=0.25*(input_tensor[i][i][0][2]+input_tensor[i][i][2][0]+ input_tensor[0][2][i][i]+input_tensor[2][0][i][i]);
            //output[4][i]=output[i][4];
          }

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][5]=0.25*(input_tensor[i][i][0][1]+input_tensor[i][i][1][0]+input_tensor[0][1][i][i]+input_tensor[1][0][i][i]);
            //output[5][i]=output[i][5];
          }

        output[3][3]=0.25*(input_tensor[1][2][1][2]+input_tensor[1][2][2][1]+input_tensor[2][1][1][2]+input_tensor[2][1][2][1]);
        output[4][4]=0.25*(input_tensor[0][2][0][2]+input_tensor[0][2][2][0]+input_tensor[2][0][0][2]+input_tensor[2][0][2][0]);
        output[5][5]=0.25*(input_tensor[1][0][1][0]+input_tensor[1][0][0][1]+input_tensor[0][1][1][0]+input_tensor[0][1][0][1]);

        output[3][4]=0.125*(input_tensor[1][2][0][2]+input_tensor[1][2][2][0]+input_tensor[2][1][0][2]+input_tensor[2][1][2][0]+input_tensor[0][2][1][2]+input_tensor[0][2][2][1]+input_tensor[2][0][1][2]+input_tensor[2][0][2][1]);
        //output[4][3]=output[3][4];
        output[3][5]=0.125*(input_tensor[1][2][0][1]+input_tensor[1][2][1][0]+input_tensor[2][1][0][1]+input_tensor[2][1][1][0]+input_tensor[0][1][1][2]+input_tensor[0][1][2][1]+input_tensor[1][0][1][2]+input_tensor[1][0][2][1]);
        //output[5][3]=output[3][5];
        output[4][5]=0.125*(input_tensor[0][2][0][1]+input_tensor[0][2][1][0]+input_tensor[2][0][0][1]+input_tensor[2][0][1][0]+input_tensor[0][1][0][2]+input_tensor[0][1][2][0]+input_tensor[1][0][0][2]+input_tensor[1][0][2][0]);
        //output[5][4]=output[4][5];

        return output;
      }

      template<int dim>
      Tensor<4,3>
      LpoSsTensor<dim>::transform_6x6_matrix_to_4th_order_tensor(const SymmetricTensor<2,6> &input_tensor)
      {
        Tensor<4,3> output;

        for (unsigned short int i = 0; i < 3; i++)
          for (unsigned short int j = 0; j < 3; j++)
            for (unsigned short int k = 0; k < 3; k++)
              for (unsigned short int l = 0; l < 3; l++)
                {
                  // The first part of the inline if statment gets the diagonal.
                  // The second part is never higher then 5 (which is the limit of the tensor index)
                  // because to reach this part the variables need to be different, which results in
                  // at least a minus 1.
                  const unsigned short int p = (i == j ? i : 6 - i - j);
                  const unsigned short int q = (k == l ? k : 6 - k - l);
                  output[i][j][k][l] = input_tensor[p][q];
                }
        return output;
      }

      template<int dim>
      Tensor<1,21>
      LpoSsTensor<dim>::transform_6x6_matrix_to_21D_vector(const SymmetricTensor<2,6> &input)
      {
        return Tensor<1,21,double> (
        {
          input[0][0],           // 0  // 1
          input[1][1],           // 1  // 2
          input[2][2],           // 2  // 3
          sqrt(2)*input[1][2],   // 3  // 4
          sqrt(2)*input[0][2],   // 4  // 5
          sqrt(2)*input[0][1],   // 5  // 6
          2*input[3][3],         // 6  // 7
          2*input[4][4],         // 7  // 8
          2*input[5][5],         // 8  // 9
          2*input[0][3],         // 9  // 10
          2*input[1][4],         // 10 // 11
          2*input[2][5],         // 11 // 12
          2*input[2][3],         // 12 // 13
          2*input[0][4],         // 13 // 14
          2*input[1][5],         // 14 // 15

          2*input[1][3],         // 15 // 16
          2*input[2][4],         // 16 // 17
          2*input[0][5],         // 17 // 18
          2*sqrt(2)*input[4][5], // 18 // 19
          2*sqrt(2)*input[3][5], // 19 // 20
          2*sqrt(2)*input[3][4]  // 20 // 21
        });

      }


      template<int dim>
      SymmetricTensor<2,6>
      LpoSsTensor<dim>::transform_21D_vector_to_6x6_matrix(const Tensor<1,21> &input)
      {
        SymmetricTensor<2,6> result;

        const double sqrt_2_inv = 1/sqrt(2);

        result[0][0] = input[0];
        result[1][1] = input[1];
        result[2][2] = input[2];
        result[1][2] = sqrt_2_inv * input[3];
        result[0][2] = sqrt_2_inv * input[4];
        result[0][1] = sqrt_2_inv * input[5];
        result[3][3] = 0.5 * input[6];
        result[4][4] = 0.5 * input[7];
        result[5][5] = 0.5 * input[8];
        result[0][3] = 0.5 * input[9];
        result[1][4] = 0.5 * input[10];
        result[2][5] = 0.5 * input[11];
        result[2][3] = 0.5 * input[12];
        result[0][4] = 0.5 * input[13];
        result[1][5] = 0.5 * input[14];
        result[1][3] = 0.5 * input[15];
        result[2][4] = 0.5 * input[16];
        result[0][5] = 0.5 * input[17];
        result[4][5] = 0.5 * sqrt_2_inv * input[18];
        result[3][5] = 0.5 * sqrt_2_inv * input[19];
        result[3][4] = 0.5 * sqrt_2_inv * input[20];

        return result;

      }


      template<int dim>
      Tensor<1,21>
      LpoSsTensor<dim>::transform_4th_order_tensor_to_21D_vector(const Tensor<4,3> &input_tensor)
      {
        return Tensor<1,21,double> (
        {
          input_tensor[0][0][0][0],           // 0  // 1
          input_tensor[1][1][1][1],           // 1  // 2
          input_tensor[2][2][2][2],           // 2  // 3
          sqrt(2)*0.5*(input_tensor[1][1][2][2] + input_tensor[2][2][1][1]),   // 3  // 4
          sqrt(2)*0.5*(input_tensor[0][0][2][2] + input_tensor[2][2][0][0]),   // 4  // 5
          sqrt(2)*0.5*(input_tensor[0][0][1][1] + input_tensor[1][1][0][0]),   // 5  // 6
          0.5*(input_tensor[1][2][1][2]+input_tensor[1][2][2][1]+input_tensor[2][1][1][2]+input_tensor[2][1][2][1]),         // 6  // 7
          0.5*(input_tensor[0][2][0][2]+input_tensor[0][2][2][0]+input_tensor[2][0][0][2]+input_tensor[2][0][2][0]),         // 7  // 8
          0.5*(input_tensor[1][0][1][0]+input_tensor[1][0][0][1]+input_tensor[0][1][1][0]+input_tensor[0][1][0][1]),         // 8  // 9
          0.5*(input_tensor[0][0][1][2]+input_tensor[0][0][2][1]+input_tensor[1][2][0][0]+input_tensor[2][1][0][0]),         // 9  // 10
          0.5*(input_tensor[1][1][0][2]+input_tensor[1][1][2][0]+input_tensor[0][2][1][1]+input_tensor[2][0][1][1]),         // 10 // 11
          0.5*(input_tensor[2][2][0][1]+input_tensor[2][2][1][0]+input_tensor[0][1][2][2]+input_tensor[1][0][2][2]),         // 11 // 12
          0.5*(input_tensor[2][2][1][2]+input_tensor[2][2][2][1]+input_tensor[1][2][2][2]+input_tensor[2][1][2][2]),         // 12 // 13
          0.5*(input_tensor[0][0][0][2]+input_tensor[0][0][2][0]+input_tensor[0][2][0][0]+input_tensor[2][0][0][0]),         // 13 // 14
          0.5*(input_tensor[1][1][0][1]+input_tensor[1][1][1][0]+input_tensor[0][1][1][1]+input_tensor[1][0][1][1]),         // 14 // 15
          0.5*(input_tensor[1][1][1][2]+input_tensor[1][1][2][1]+input_tensor[1][2][1][1]+input_tensor[2][1][1][1]),         // 15 // 16
          0.5*(input_tensor[2][2][0][2]+input_tensor[2][2][2][0]+input_tensor[0][2][2][2]+input_tensor[2][0][2][2]),         // 16 // 17
          0.5*(input_tensor[0][0][0][1]+input_tensor[0][0][1][0]+input_tensor[0][1][0][0]+input_tensor[1][0][0][0]),         // 17 // 18
          sqrt(2)*0.25*(input_tensor[0][2][0][1]+input_tensor[0][2][1][0]+input_tensor[2][0][0][1]+input_tensor[2][0][1][0]+input_tensor[0][1][0][2]+input_tensor[0][1][2][0]+input_tensor[1][0][0][2]+input_tensor[1][0][2][0]), // 18 // 19
          sqrt(2)*0.25*(input_tensor[1][2][0][1]+input_tensor[1][2][1][0]+input_tensor[2][1][0][1]+input_tensor[2][1][1][0]+input_tensor[0][1][1][2]+input_tensor[0][1][2][1]+input_tensor[1][0][1][2]+input_tensor[1][0][2][1]), // 19 // 20
          sqrt(2)*0.25*(input_tensor[1][2][0][2]+input_tensor[1][2][2][0]+input_tensor[2][1][0][2]+input_tensor[2][1][2][0]+input_tensor[0][2][1][2]+input_tensor[0][2][2][1]+input_tensor[2][0][1][2]+input_tensor[2][0][2][1])  // 20 // 21
        });

      }

      template<int dim>
      std::vector<Tensor<2,3> >
      LpoSsTensor<dim>::random_draw_volume_weighting(std::vector<double> fv,
                                                          std::vector<Tensor<2,3>> matrices,
                                                          unsigned int n_output_grains) const
      {
        // Get volume weighted euler angles, using random draws to convert odf
        // to a discrete number of orientations, weighted by volume
        // 1a. Get index that would sort volume fractions AND
        //ix = np.argsort(fv[q,:]);
        // 1b. Get the sorted volume and angle arrays
        std::vector<double> fv_to_sort = fv;
        std::vector<double> fv_sorted = fv;
        std::vector<Tensor<2,3>> matrices_sorted = matrices;

        unsigned int n_grain = fv_to_sort.size();


        /**
         * ...
         */
        for (int i = n_grain-1; i >= 0; --i)
          {
            unsigned int index_max_fv = std::distance(fv_to_sort.begin(),max_element(fv_to_sort.begin(), fv_to_sort.end()));

            fv_sorted[i] = fv_to_sort[index_max_fv];
            matrices_sorted[i] = matrices[index_max_fv];
            /*Assert(matrices[index_max_fv].size() == 3, ExcMessage("matrices vector (size = " + std::to_string(matrices[index_max_fv].size()) +
                                                                ") should have size 3."));
            Assert(matrices_sorted[i].size() == 3, ExcMessage("matrices_sorted vector (size = " + std::to_string(matrices_sorted[i].size()) +
                                                            ") should have size 3."));*/
            fv_to_sort[index_max_fv] = -1;
          }

        // 2. Get cumulative weight for volume fraction
        std::vector<double> cum_weight(n_grains);
        std::partial_sum(fv_sorted.begin(),fv_sorted.end(),cum_weight.begin());
        // 3. Generate random indices
        boost::random::uniform_real_distribution<> dist(0, 1);
        std::vector<double> idxgrain(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            idxgrain[grain_i] = dist(this->random_number_generator);
          }

        // 4. Find the maximum cum_weight that is less than the random value.
        // the euler angle index is +1. For example, if the idxGrain(g) < cumWeight(1),
        // the index should be 1 not zero)
        std::vector<Tensor<2,3>> matrices_out(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            unsigned int counter = 0;
            for (unsigned int grain_j = 0; grain_j < n_grains; ++grain_j)
              {
                // find the first cummulative weight which is larger than the random number
                // todo: there are algorithms to do this faster
                if (cum_weight[grain_j] < idxgrain[grain_i])
                  {
                    counter++;
                  }
                else
                  {
                    break;
                  }


                /*Assert(matrices_sorted[counter].size() == 3, ExcMessage("matrices_sorted vector (size = " + std::to_string(matrices_sorted[counter].size()) +
                                                                      ") should have size 3."));*/

                /*Assert(matrices_out[counter].size() == 3, ExcMessage("matrices_out vector (size = " + std::to_string(matrices_out[counter].size()) +
                                                                   ") should have size 3."));*/
              }
            matrices_out[grain_i] = matrices_sorted[counter];
          }
        return matrices_out;
      }



      template <int dim>
      UpdateTimeFlags
      LpoSsTensor<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      LpoSsTensor<dim>::get_needed_update_flags () const
      {
        return update_default;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      LpoSsTensor<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int> > property_information;

        property_information.push_back(std::make_pair("lpo_Ss_tensor",Tensor<2,6>::n_independent_components));

        return property_information;
      }

      template <int dim>
      void
      LpoSsTensor<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoSsTensor");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducable as long as the problem is run with the "
                                 "same amount of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");


              prm.declare_entry ("Volume fraction olivine", "0.5",
                                 Patterns::Double(0),
                                 "The volume fraction of the olivine phase (0 is no olivine, 1 is fully olivine). "
                                 "The rest of the volume fraction is set to be entstatite. "
                                 "Todo: if full olivine make not enstite grains and vice-versa.");

              prm.declare_entry ("Number of samples", "0",
                                 Patterns::Double(0),
                                 "This determines how many samples are taken when using the random "
                                 "draw volume averaging. Setting it to zero means that the number of "
                                 "samples is set to be equal to the number of grains.");
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }


      template <int dim>
      void
      LpoSsTensor<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoSsTensor");
            {

              random_number_seed = prm.get_integer ("Random number seed"); // 2
              n_grains = LPO<dim>::get_number_of_grains();
              n_minerals = LPO<dim>::get_number_of_minerals();
              n_samples = prm.get_integer("Number of samples"); // 0
              if (n_samples == 0)
                n_samples = n_grains;
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();


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
      ASPECT_REGISTER_PARTICLE_PROPERTY(LpoSsTensor,
                                        "lpo Ss tensor",
                                        "A plugin in which the particle property tensor is "
                                        "defined as he collection of stresses resulted from "
                                        "the micromechanical model for olivine aggregate deformation "
                                        "with the current strain rate and 5 independent strain rates "
                                        "with the same amplitude. These stresses can be used to construct "
                                        "the full rank4 viscosity tensor.")
    }
  }
}