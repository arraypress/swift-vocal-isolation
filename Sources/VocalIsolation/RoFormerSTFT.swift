//
//  RoFormerSTFT.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//
//  BS-RoFormer's spectrogram and its way back, in vDSP: torch.stft with
//  n_fft 2048, hop 441, a periodic Hann window, centre reflect padding and no
//  normalisation, laid out per frame as (bin, channel, real/imaginary) — the
//  'b t (f s c)' the band split reads — and torch.istft of the masked
//  spectrogram with the DC bin zeroed, as upstream's forward does after the
//  mask. The layouts are upstream's; nothing is re-authored.
//

import Accelerate
import Foundation

/// The spectrogram BS-RoFormer sees, and the inverse of its masked output.
final class RoFormerSTFT {

    static let nFFT = 2048
    static let hop = 441
    static let bins = nFFT / 2 + 1   // 1025, Nyquist kept
    static let channels = 2
    /// Values per frame in the model's layout: bins × channels × (re, im).
    static let width = bins * channels * 2

    private let window: [Float]
    private let windowSq: [Float]
    private let fwd: vDSP_DFT_Setup
    private let inv: vDSP_DFT_Setup
    private var envelopes: [Int: [Float]] = [:]

    init?() {
        var w = [Float](repeating: 0, count: Self.nFFT)
        for n in 0..<Self.nFFT { w[n] = Float(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(Self.nFFT))) }
        window = w
        windowSq = w.map { $0 * $0 }
        guard let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD),
              let i = vDSP_DFT_zop_CreateSetup(f, vDSP_Length(Self.nFFT), .INVERSE) else { return nil }
        fwd = f; inv = i
    }

    deinit { vDSP_DFT_DestroySetup(inv); vDSP_DFT_DestroySetup(fwd) }

    /// 1 + samples / hop, torch.stft's frame count with centre padding.
    static func frames(forSamples count: Int) -> Int { 1 + count / hop }

    // MARK: - Forward

    /// `[frames][width]`: for each frame, bins × (L, R) × (re, im).
    func spectrogram(_ channels: [[Float]]) -> (values: [Float], frames: Int) {
        let length = channels[0].count
        let frames = Self.frames(forSamples: length)
        var out = [Float](repeating: 0, count: frames * Self.width)
        var frameReal = [Float](repeating: 0, count: Self.nFFT)
        let frameImag = [Float](repeating: 0, count: Self.nFFT)
        var outR = [Float](repeating: 0, count: Self.nFFT)
        var outI = [Float](repeating: 0, count: Self.nFFT)
        out.withUnsafeMutableBufferPointer { op in
            for ch in 0..<Self.channels {
                let padded = DemucsSTFT.reflectPadded(channels[ch], left: Self.nFFT / 2, right: Self.nFFT / 2)
                padded.withUnsafeBufferPointer { pp in
                    for t in 0..<frames {
                        vDSP_vmul(pp.baseAddress! + t * Self.hop, 1, window, 1, &frameReal, 1, vDSP_Length(Self.nFFT))
                        vDSP_DFT_Execute(fwd, frameReal, frameImag, &outR, &outI)
                        let base = t * Self.width
                        for f in 0..<Self.bins {
                            let k = base + (f * Self.channels + ch) * 2
                            op[k] = outR[f]; op[k + 1] = outI[f]
                        }
                    }
                }
            }
        }
        return (out, frames)
    }

    // MARK: - Inverse

    private func envelope(frames: Int, paddedLen: Int) -> [Float] {
        if let cached = envelopes[frames] { return cached }
        var envelope = [Float](repeating: 0, count: paddedLen)
        envelope.withUnsafeMutableBufferPointer { ep in
            for t in 0..<frames {
                vDSP_vadd(ep.baseAddress! + t * Self.hop, 1, windowSq, 1, ep.baseAddress! + t * Self.hop, 1, vDSP_Length(Self.nFFT))
            }
        }
        envelopes[frames] = envelope
        return envelope
    }

    /// Upstream's tail for one stem: complex mask × spectrogram (same `[frames][width]`
    /// layout), DC bin zeroed, `torch.istft(length:)` → stereo of `length` samples.
    func inverse(spectrogram spec: [Float], mask: [Float], frames: Int, length: Int, zeroDC: Bool = true) -> [[Float]] {
        precondition(spec.count == frames * Self.width && mask.count == frames * Self.width)
        let paddedLen = (frames - 1) * Self.hop + Self.nFFT
        let centre = Self.nFFT / 2
        precondition(centre + length <= paddedLen)
        let env = envelope(frames: frames, paddedLen: paddedLen)
        let scale = 1 / Float(Self.nFFT)          // the DFT's 1/N; normalized=False
        var scaledWindow = [Float](repeating: 0, count: Self.nFFT)
        var s = scale
        vDSP_vsmul(window, 1, &s, &scaledWindow, 1, vDSP_Length(Self.nFFT))
        var out: [[Float]] = []
        var fullR = [Float](repeating: 0, count: Self.nFFT)
        var fullI = [Float](repeating: 0, count: Self.nFFT)
        var timeR = [Float](repeating: 0, count: Self.nFFT)
        var timeI = [Float](repeating: 0, count: Self.nFFT)
        var tmp = [Float](repeating: 0, count: Self.nFFT)
        spec.withUnsafeBufferPointer { sp in
            mask.withUnsafeBufferPointer { mp in
                for ch in 0..<Self.channels {
                    var recon = [Float](repeating: 0, count: paddedLen)
                    recon.withUnsafeMutableBufferPointer { rp in
                        for t in 0..<frames {
                            let base = t * Self.width
                            for f in 0..<Self.bins {
                                let k = base + (f * Self.channels + ch) * 2
                                // (a + bi)(c + di)
                                let a = sp[k], b = sp[k + 1], c = mp[k], d = mp[k + 1]
                                fullR[f] = a * c - b * d
                                fullI[f] = a * d + b * c
                            }
                            if zeroDC { fullR[0] = 0; fullI[0] = 0 }
                            for k in Self.bins..<Self.nFFT { fullR[k] = fullR[Self.nFFT - k]; fullI[k] = -fullI[Self.nFFT - k] }
                            vDSP_DFT_Execute(inv, fullR, fullI, &timeR, &timeI)
                            vDSP_vmul(timeR, 1, scaledWindow, 1, &tmp, 1, vDSP_Length(Self.nFFT))
                            vDSP_vadd(rp.baseAddress! + t * Self.hop, 1, tmp, 1, rp.baseAddress! + t * Self.hop, 1, vDSP_Length(Self.nFFT))
                        }
                    }
                    var channel = [Float](repeating: 0, count: length)
                    recon.withUnsafeBufferPointer { rp in
                        env.withUnsafeBufferPointer { ep in
                            vDSP_vdiv(ep.baseAddress! + centre, 1, rp.baseAddress! + centre, 1, &channel, 1, vDSP_Length(length))
                        }
                    }
                    out.append(channel)
                }
            }
        }
        return out
    }
}
