//
//  Submission.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftData

struct Submission : Codable {
    var submission_id: String
    var user_id: String
    var contest_id: String
    var submitted_at: String
    var vote_count: Int
}
