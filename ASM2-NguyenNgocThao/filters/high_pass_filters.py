from filters.low_pass_filters import ideal_lowpass_filter, gaussian_lowpass_filter, butterworth_lowpass_filter


def ideal_highpass_filter(size, cutoff):
    return 1 - ideal_lowpass_filter(size, cutoff)


def gaussian_highpass_filter(size, cutoff):
    return 1 - gaussian_lowpass_filter(size, cutoff)


def butterworth_highpass_filter(size, cutoff, order=2):
    return 1 - butterworth_lowpass_filter(size, cutoff, order)
