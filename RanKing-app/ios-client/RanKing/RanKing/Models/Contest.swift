//
//  Contest.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/12/25.
//

import Foundation
import SwiftData

final class Contest : Codable  {
    var contest_id : Int
    var name: String
    var description: String
    var submission_start_date: String
    var submission_end_date: String
    var voting_end_date : String
}
