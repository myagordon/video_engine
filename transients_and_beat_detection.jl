import Pkg
#packages
Pkg.add.(["WAV", "DSP", "FFTW", "Statistics", "Plots"]) 

#imoprts
using WAV #read in WAV
using DSP #fitlering, resampling, window stuff
using FFTW #does FFT
using Statistics
using Plots

#constants
const FS = 44100
const N = 1024 #STFT window len in samples
const H = 256 #hop size in samples

#small Hann Window
hann(N::Int) = 0.5f0 .- 0.5f0 .* cos.(2f0*π .* (0:N-1) ./ (N-1))

#load audio 
function load_audio_mono(path::AbstractString; fs::Int=FS)
    y, fs = wavread(path) #y tells you how many channels it samples?
    y = ndims(y) == 1 ? reshape(y, :, 1) : y # makes sure the audio array is 2D
    x_mono = Float32.(vec(mean(y,dims=2))) # avgs channels to get mono signal, float32
    return x_mono, Int(round(fs))   # ensure sample rate is Int
end

function frame_signal(x::Vector{Float32}, N::Int, H::Int)
    W = hann(N)
    #overlapping frames with hop size H
    L = length(x)
    nframes = L < N ? 0 : 1 + div(L - N, H)
    X = Array{Float32}(undef, N, nframes)
    @inbounds for i = 1:nframes
        s = (i-1)*H + 1
        X[:, i] = x[s:s+N-1] .* W
    end
    return X #return an N point Hann Window
end

function stft_mag(x::Vector{Float32}, N::Int, H::Int)
    Xw = frame_signal(x,N,H)
    nframes = size(Xw, 2)
    nfreq = div(N,2) + 1
    S = Array{Float32}(undef, nfreq, nframes)
    for i = 1:nframes
        #compute a real fft each frame, collecting magnitudes in an array
        S[:, i] = Float32.(abs.(rfft(Xw[:, i])))
    end
    return S
end

#Slog = log1p(α * |X|) log compression
log_compress(S::AbstractMatrix{<:Real}; alpha::Float32=5.0f0) = log1p.(alpha .* S)

function spectral_flux(Slog::AbstractMatrix{<:Real})
    nfreq, nframes = size(Slog)
    if nframes < 2
        return zeros(Float32, nframes)
    end
    Δ = Slog[:, 2:end] .- Slog[:, 1:end-1] # temporal diff
    Δpos = max.(Float32.(Δ), 0f0) # half wave rectify (keep positives)
    flux = vec(sum(Δpos; dims=1)) # sum over freqs so 1×(nFrames-1)
    flux = vcat(flux, 0f0) # pad to nFrames
    return flux
end

# ema smoothing
function ema(x; alpha::Float32=0.3f0)
    y= similar(Float32.(x))
    y[1] = Float32(x[1])
    @inbounds for n in 2:length(x)
        y[n] = (1f0-alpha)*y[n-1] + alpha*Float32(x[n])
    end
    y
end

#moving local percentile baseline
function moving_percentile(x, win_frames::Int; q::Float64=0.6)
    L = length(x); half = max(win_frames,1) ÷ 2
    b = zeros(Float32,L)
    @inbounds for n in 1:L
        a = max(1,n-half); bnd = min(L,n+half)
        b[n] = Float32(quantile(@view(x[a:bnd]), q))
    end
    b
end

#moving median baseline
function moving_median(x, win_frames::Int)
    L = length(x); half = max(win_frames,1) ÷ 2
    b = zeros(Float32, L)
    @inbounds for n in 1:L
        a = max(1,n-half); bnd = min(L,n+half)
        b[n] = Float32(median(@view(x[a:bnd])))
    end
    b
end

#peak picking
#1. find local max
#2. find all peaks above baseline + delta
#3. enforce refractory period by keeping highest peak per window
function pick_peaks(flux, base; delta::Float32=0f0, refractory_ms::Int=100, fs::Real=FS, hop::Int=H)
    L = length(flux)
    # find all possible peaks within window 
    cand = Int[]
    @inbounds for n in 2:L-1
        if flux[n] > base[n] + delta && flux[n] > flux[n-1] && flux[n] >= flux[n+1]
            push!(cand, n)
        end
    end
    # refractory period
    refr = max(1, Int(ceil(refractory_ms/1000 * fs / hop)))
    keep = Int[]; i = 1
    while i <= length(cand)
        j = i; best = cand[i]
        while j+1 <= length(cand) && cand[j+1] - cand[i] < refr
            if flux[cand[j+1]] > flux[best]
                best = cand[j+1]
            end
            j += 1
        end
        push!(keep, best)
        i = j + 1
    end
    # simple 0 to 1 confidence
    conf = Float32.(flux[keep] .- base[keep])
    m = maximum(conf; init=0f0); if m > 0f0; conf ./= m; end
    return keep, conf
end

#=
# simple spectrogram plotter for log magnitude form
function simple_spectrogram(Slog, fs; alpha=5.0)
    # time / freq axes
    t = (0:size(Slog,2)-1) .* (H / fs)
    f = (0:div(N,2)) .* (fs / N)

    heatmap(t, f, Slog;
        xlabel="Time (s)", ylabel="Frequency (Hz)",
        title="Log magnitude Spectrogram,  a=$(alpha))",
        ylims=(0, min(8000, fs/2)))
end

function plot_flux(flux::AbstractVector{<:Real}, fs::Real)
    t = (0:length(flux)-1) .* (H / fs)
    plot(t, flux; xlabel="Time (s)", ylabel="Spectral Flux",
         title="Spectral Flux", color=:black)
end
=#

# bpm detection functions

# frames/sec for flux signal
frames_per_sec(fs::Real, hop::Integer) = fs / hop

# Compute autocorr at integer increments 
@views function autocorr_at_lags(x::AbstractVector{<:Real}, lags::UnitRange{Int})
    xz = Float32.(x .- mean(x))
    st_dev  = std(xz); st_dev = st_dev == 0 ? 1f0 : st_dev
    xz ./= st_dev
    L = length(xz)
    c = Vector{Float32}(undef, length(lags))
    for (i, l) in enumerate(lags)
        if l < L
            n = L - l
            c[i] = sum(xz[1:n] .* xz[1+l:L]) / n
        else
            c[i] = 0f0
        end
    end
    return c
end


# Estimate one global BPM from flux
function estimate_bpm_from_flux(flux::Vector{<:Real}; fs::Real=FS, hop::Integer=H,
                                bpm_lo::Float64=80.0, bpm_hi::Float64=180.0)
    fps = frames_per_sec(fs, hop)
    lag_lo = Int(floor(fps * 60 / bpm_hi))
    lag_hi = Int(ceil(  fps * 60 / bpm_lo))
    lag_lo = max(lag_lo, 2)
    lag_hi = min(lag_hi, length(flux) ÷ 2)
    lags = lag_lo:lag_hi
    ac = autocorr_at_lags(flux, lags)
    i_max = argmax(ac)
    best_lag = lags[i_max]
    bpm = (fps * 60) / best_lag
    return bpm, best_lag, ac, lags
end

# Pick t0 using first strong onset in secs
function pick_phase_zero_from_onsets(peaks::Vector{Int}; fs::Real=FS, hop::Integer=H)
    isempty(peaks) && return 0.0
    n0 = peaks[1]
    return (n0-1) * hop / fs
end

function plot_flux_overlay(flux, base, peaks, fs)
    t = (0:length(flux)-1) .* (H / fs)
    plot(t, flux; label="flux (EMA)", color=:black, xlabel="Time (s)", ylabel="Value",
         title="Flux, baseline, and peaks")
    plot!(t, base; label="baseline", color=:red, alpha=0.8)
    scatter!(t[peaks], flux[peaks]; label="peaks", color=:blue, ms=4)
end

#main 
function main()
    cd(@__DIR__)
    x, fsfile = load_audio_mono("dread_architect.wav")

    S     = stft_mag(x, N, H)
    Slog  = log_compress(S; alpha=5.0f0)
    flux  = spectral_flux(Slog)

    flux_s = ema(flux; alpha=0.3f0) # smooth
    win_s  = 0.6  # baseline window in secs
    win_fr = max(3, Int(ceil(win_s * fsfile / H)))

    base = moving_percentile(flux_s, win_fr; q=0.6) # or: base = moving_median(flux_s, win_fr)
    #gets peak every 3 seconds, can adjust this if we want
    peaks, conf = pick_peaks(flux_s, base; delta=0.0f0, refractory_ms=3000, fs=fsfile, hop=H)

    plot_flux_overlay(flux_s, base, peaks, fsfile)
    savefig("flux_baseline_peaks.png")
    println("Saved: flux_baseline_peaks image")

    # Estimate BPM globally
    bpm, best_lag, ac, lags = estimate_bpm_from_flux(flux; fs=fsfile, hop=H,
                                                 bpm_lo=80.0, bpm_hi=180.0)
    @show bpm
    # Choose phase zero from first onset
    t0 = pick_phase_zero_from_onsets(peaks; fs=fsfile, hop=H)
    @show t0
    # Visual check: autocorr vs lags (frames) and mapped BPM
    # Plot BPM for debugging 
    p1 = plot(lags, ac, xlabel="Lag (frames)", ylabel="Autocorr", title="Flux autocorr")
    bpm_axis = (frames_per_sec(fsfile,H) * 60) ./ lags
    p2 = plot(bpm_axis, ac, xlabel="BPM (mapped)", ylabel="Autocorr", title="Tempo pick")
    display(plot(p1, p2, layout=(1,2), size=(900,350)))

end

main()
