//
//  VotingContest.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/12/25.
//

import Foundation
import SwiftData

@Model
final class VotingContest : Identifiable {
    var name: String
    
    init(name: String) {
        self.name = name
    }
}
