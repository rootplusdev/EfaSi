import numpy as np


def get_total_num_encoding(line_length: int) -> int:
    """Get total number of encoding for a line of given length."""
    from line_encoding_cpp import get_total_num_encoding

    return get_total_num_encoding(line_length)


def get_encoding_usage_flags(line_length: int) -> np.ndarray:
    """
    Get encoding usage flags of a encoding map.
    Returns: int8 np.ndarray of shape (total_num_encoding,)
    """
    from line_encoding_cpp import get_total_num_encoding, get_encoding_usage_flag

    total_num_encoding = get_total_num_encoding(line_length)
    usage_flags = np.zeros(total_num_encoding, dtype=np.int8)
    get_encoding_usage_flag(usage_flags, line_length)

    return usage_flags


def transform_lines_to_line_encoding(lines_input: np.ndarray, 
                                     line_length: int) -> np.ndarray:
    """
    Get line encoding of the given batched lines.
    Args:
        lines_input: Lines int8 array of shape [N, L]. 
            Elements are in {0,1,2} for empty/self/oppo.
        line_length: Length of line encoding.
    Returns:
        line_encodings: Line encoding int32 array of shape [N, L].
    """
    from line_encoding_cpp import transform_lines_to_line_encoding
    assert lines_input.ndim == 2 and lines_input.dtype == np.int8
    assert np.min(lines_input) >= 0 and np.max(lines_input) <= 2
    
    line_encodings_output = np.zeros_like(lines_input, dtype=np.int32)
    transform_lines_to_line_encoding(lines_input, line_encodings_output, line_length)
    
    return line_encodings_output
