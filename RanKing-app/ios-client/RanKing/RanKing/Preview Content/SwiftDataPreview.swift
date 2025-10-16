//
//  SwiftDataPreview.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/12/25.
//

import SwiftData
import SwiftUI

class SampleData {
    static let shared = SampleData() // Singleton instance

    let modelContainer: ModelContainer

    private init() {
        let schema = Schema([
            Contest.self // Add all your SwiftData models here
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)

        do {
            modelContainer = try ModelContainer(for: schema, configurations: [modelConfiguration])
            // Add sample data to the container
            Task { @MainActor in
                await addSampleData()
            }
        } catch {
            fatalError("Could not create ModelContainer for preview: \(error)")
        }
    }

    @MainActor
    private func addSampleData() async {
        let context = modelContainer.mainContext

        // Create and insert sample data here
        let item1 = Contest(name: "Sample Contest", contestDescription: "This is a sample contest.", startDate: Date() , endDate: Date(), contestPhase: ContestPhase.SUBMISSION)
        context.insert(item1)
        
        let item2 = Contest(name: "Sample Contest 2", contestDescription: "This is a sample contest number 2", startDate: Date() , endDate: Date(), contestPhase: ContestPhase.SUBMISSION)
        context.insert(item2)
        
        let item3 = Contest(name: "Sample Contest Active", contestDescription: "This is a sample contest number 2", startDate: Date() , endDate: Date(), contestPhase: ContestPhase.VOTING)
        context.insert(item3)
        
        let item4 = Contest(name: "Sample Contest Active 2", contestDescription: "This is a sample contest number 2", startDate: Date() , endDate: Date(), contestPhase: ContestPhase.VOTING)
        context.insert(item4)
        
        do {
            try context.save()
        } catch {
            print("Error saving sample data: \(error)")
        }
    }
}
