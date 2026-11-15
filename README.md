# Compression-Lib

[![CI](https://github.com/kapildhinwa2010-dev/Compression-Lib/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Lib/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Compression-Lib/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Lib/actions/workflows/codeql.yml)

Reusable compression-codec library for 2-D float matrices.

## What it does

Four codecs - SVD low-rank, uniform quantization, Fourier-basis fitting,
and a hybrid (SVD then quantize) - plus storage estimates and
best_method(), which returns the smallest codec that meets a
relative-error budget.

    from compression_lib import best_method

    pick = best_method(matrix, max_error=0.05)
    if pick:
        print(pick["method"], pick["compression_ratio"])

## API

- svd_compress(matrix, rank), quantize(matrix, bits),
  fourier_compress(matrix, num_coeffs), hybrid_compress(matrix, rank, bits)
- estimated_storage_bytes(matrix, method), compression_ratio(matrix, method)
- best_method(matrix, max_error) - dict or None
- fourier_basis(n_points, num_coeffs) - the Fourier design matrix

## Honest results and limitations

- Real gains only on low-rank or smooth (band-limited) matrices. On
  random Gaussian matrices every codec has relative error near 1.0 and
  best_method() returns None - that is the expected, honest outcome.
- Storage estimates are param counts, not a real serialized format;
  actual on-disk size would add headers and dtype metadata.
- quantize() is lossy by design; bits below 6 visibly distort most data.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
