//
//  User.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/16/25.
//

import Foundation

public struct User: Codable {
    public var userId: Int
    public var username: String
    public var email: String
    public var createdAt: String
}
