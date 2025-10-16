//
//  Contest.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/12/25.
//

import Foundation
import SwiftData

@Model
final class Contest : Identifiable {
    var id : UUID
    var name: String
    var contestDescription: String
    var numSumbissions: Int
    var startDate: Date
    var endDate: Date
    var contestPhase : ContestPhase
    
    init(name: String, contestDescription: String, startDate: Date, endDate: Date, contestPhase : ContestPhase) {
        self.id = UUID()
        self.name = name
        self.contestDescription = contestDescription
        self.numSumbissions = 0
        self.startDate = startDate
        self.endDate = endDate
        self.contestPhase = contestPhase
    }
}

enum ContestPhase : String, Codable {
    case SUBMISSION = "submission"
    case VOTING = "voting"
    case FINISHED = "finished"
}
