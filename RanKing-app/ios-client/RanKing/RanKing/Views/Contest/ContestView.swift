//
//  ContestView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI

struct ContestView: View {
    //TODO polish this
    var body: some View {
        Text("Best Contest out There")
            .font(.largeTitle)
            .padding(.top)
        Text("Dress nicely and see if you win the contest")
        Spacer()
        Text("Submissions: {}")
            .font(.headline)
        Text("Voting Period: Oct 10, 2025 - Oct 17, 2025")
        VStack {
            Text("Submit Fashion Image")
                .font(.headline)
                .padding()
            Button("Upload Image") {
                // TODO: implement image picker + API upload
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

#Preview {
    ContestView()
}
