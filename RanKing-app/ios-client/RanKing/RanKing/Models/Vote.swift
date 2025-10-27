//
//  Vote.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/16/25.
//

import Foundation

public struct Vote: Codable {
    public var voteId: Int
    public var userId: Int
    public var submissionId: Int
    public var votedAt: String
}
