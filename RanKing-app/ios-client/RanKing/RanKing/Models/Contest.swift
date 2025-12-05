//
//  Contest.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/12/25.
//

import Foundation
import SwiftData

final class ContestResponse : Codable  {
    var statusCode: Int
    var body: String
}

final class Contest : Codable  {
    var contest_id : String
    var name: String
    var description: String
    var submission_start_date: Date
    var submission_end_date: Date
    var voting_end_date : Date
}
