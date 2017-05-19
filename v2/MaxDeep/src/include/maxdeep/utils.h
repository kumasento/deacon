/**
 * This header file contains several useful utility functions.
 *
 * In order to use this file, please remember to add.
 */
#ifndef MAXDEEP_UTILS_H__
#define MAXDEEP_UTILS_H__

#include <string>

namespace maxdeep {

namespace utils {

/**
 * Get number of elements aligned by memory burst size.
 *
 * @param size number of elements of a given array.
 * @param num_bytes number of bytes of each array element.
 * @param burst_size number of bytes within a burst.
 * @return number of elements of the ALIGNED array.
 */
unsigned int burst_aligned_size(unsigned int size, size_t num_bytes, size_t burst_size) {
  return (unsigned int) (ceil((double) size * num_bytes / burst_size) * burst_size / num_bytes);
}

/**
 * Determine whether the current run is in a simulation environment.
 *
 * The idea is quite simple: search whether _sim appears in the program name.
 *
 * @param program_name name of the current program
 * @return whether the program is in a simulation environment
 */
bool is_sim(std::string program_name) {
  return program_name.find(std::string("_sim")) != std::string::npos;
}
  
}

}

#endif
