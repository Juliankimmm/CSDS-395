//
//  Submission.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftData

struct Submission : Codable {
    var submission_id: Int
    var user_id: Int
    var contest_id: Int
    var image_path: String
    var submitted_at: String
    var vote_count: Int
    var s3_key : String
    
}
