//
//  RanKingApp.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI
import SwiftData

@main
struct RanKingApp: App {
    var body: some Scene {
        WindowGroup {
            LoginView()
                .modelContainer( for: [
                    Contest.self
                ])
        }
    }
}
