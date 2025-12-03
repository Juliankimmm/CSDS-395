//
//  Submission.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftData

final class SubmissionResponse: Identifiable, Codable {
    var sub_id: Int
    var user_id: Int
    var contest_id: Int
    var image_path: String
    var submitted_at: String
    
    init() {
        sub_id = 0
        user_id = 0
        contest_id = 0
        image_path = ""
        submitted_at = ""
    }
}
