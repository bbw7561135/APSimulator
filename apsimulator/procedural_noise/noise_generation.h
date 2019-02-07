#ifdef __cplusplus
extern "C"
#endif
void generate_perlin_noise(float* noise_values,
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
                           int number_of_threads);
