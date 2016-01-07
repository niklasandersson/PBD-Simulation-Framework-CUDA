#ifndef CUDA_CALL_ERROR_H
#define CUDA_CALL_ERROR_H

#include <stdexcept>

class cuda_call_error : public std::domain_error {

public:
    cuda_call_error(const std::string message) : std::domain_error(message) {}

protected:

private:

};

#endif // CUDA_CALL_ERROR_H