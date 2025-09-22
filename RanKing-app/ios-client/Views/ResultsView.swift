import SwiftUI

struct ResultsView: View {
    var body: some View {
        VStack {
            Text("Leaderboard")
            // TODO: fetch from backend
            List {
                Text("1. Outfit XYZ")
                Text("2. Outfit ABC")
            }
        }
    }
}
