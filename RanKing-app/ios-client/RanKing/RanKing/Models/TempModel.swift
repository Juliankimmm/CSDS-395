
//
//  TempModel.swift
//  RanKing
//
//  Created by Damario Hamilton on 11/29/25.
//

import Foundation



struct UserProfile {
    var name: String
    var handle: String
    var avatar: String
    var submissions: [Submission]
    var totalVotesCast: Int
    var avgPercentile: Double
    var avgWinRate: Double
}

struct Submission: Identifiable {
    var id = UUID()
    var imageName: String
    var winRate: Double
    var votes: Int
    var rank: Int     // for best submission
}

