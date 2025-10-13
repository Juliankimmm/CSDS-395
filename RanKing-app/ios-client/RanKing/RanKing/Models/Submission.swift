//
//  Submission.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftData

@Model
final class Submission: Identifiable {
    var id: UUID
    var username: String
    var email: String
    var imagePath: String
    var contestId: Int64
    var votesAgainst: Int
    
    init(username: String, email: String, imagePath: String, contestId: Int64, votesAgainst: Int) {
        self.id = UUID()
        self.username = username
        self.email = email
        self.imagePath = imagePath
        self.contestId = contestId
        self.votesAgainst = votesAgainst
    }
}
