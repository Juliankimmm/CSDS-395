//
//  SubmissionGrid.swift
//  RanKing
//
//  Created by Damario Hamilton on 11/29/25.
//


struct SubmissionGrid: View {
    let submissions: [Submission]
    
    let columns = [
        GridItem(.flexible()), GridItem(.flexible())
    ]
    
    var body: some View {
        LazyVGrid(columns: columns, spacing: 20) {
            ForEach(submissions) { sub in
                SubmissionCard(submission: sub)
            }
        }
        .padding(.horizontal)
    }
}

struct SubmissionCard: View {
    let submission: Submission
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(submission.imageName)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(height: 180)
                .clipped()
                .cornerRadius(16)
            
            HStack {
                Text("\(Int(submission.winRate * 100))%")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                Text("\(submission.votes) votes")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
        }
    }
}
