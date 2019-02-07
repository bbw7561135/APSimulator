#include "noise_generation.h"
#include <omp.h>
#include <noise/noise.h>
#include <stdio.h>

extern "C" void generate_perlin_noise(float* noise_values,
                                      unsigned int* value_indices,
                                      float* x_coordinates,
                                      float* y_coordinates,
                                      float* z_coordinates,
                                      int number_of_coordinates,
                                      int number_of_octaves,
                                      float initial_frequency,
                                      float lacunarity,
                                      float persistence,
                                      int seed,
                                      int number_of_threads)
{
    noise::module::Perlin perlinModule;
    perlinModule.SetOctaveCount(number_of_octaves);
    perlinModule.SetFrequency(initial_frequency);
    perlinModule.SetLacunarity(lacunarity);
    perlinModule.SetPersistence(persistence);
    perlinModule.SetSeed(seed);
    perlinModule.SetNoiseQuality(noise::QUALITY_STD);

    float total_amplitude;
    float normalization;
    int idx;

    total_amplitude = 0;
    for (idx = 0; idx < number_of_octaves; idx++)
        total_amplitude = 1 + total_amplitude*persistence;

    normalization = 1/total_amplitude;

    #pragma omp parallel for num_threads(number_of_threads) schedule(static) default(shared) private(idx)
    for (idx = 0; idx < number_of_coordinates; idx++)
    {
        //printf("%u, %g, %g, %g\n", value_indices[idx], x_coordinates[idx], y_coordinates[idx], z_coordinates[idx]);
        noise_values[value_indices[idx]] = perlinModule.GetValue(x_coordinates[idx], y_coordinates[idx], z_coordinates[idx])*normalization;
    }
}
