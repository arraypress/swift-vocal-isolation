//
//  DFTSetup.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

import Accelerate

/// A vDSP DFT setup the STFTs can share. Read-only once made, and vDSP
/// documents executing one setup from several threads at once as safe; the
/// C pointer is the only reason this is unchecked.
struct DFTSetup: @unchecked Sendable {
    let raw: vDSP_DFT_Setup
}
