//
//  ContestView.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/4/25.
//

import SwiftUI
import PhotosUI


struct ContestViewData {
    let contestTitle: String
    let contestDescription: String
    let contestId: String
    let votingPeriod: DateInterval
}

struct ContestView: View {
    
    let networkManager = NetworkManager.getInstance()
    
    let contestData : ContestViewData
    
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var displayedImage: Image?
    
    var body: some View {
        VStack {
            Text(contestData.contestTitle)
                .font(.largeTitle)
                .padding(.top)
            Text(contestData.contestDescription)
            Spacer()
            
            if let displayedImage {
                displayedImage
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .shadow(radius: 5)
                    .padding()
                Button("Submit") {
                    Task {
                        if let data = try? await selectedImageItem?.loadTransferable(type: Data.self) {
                            await postImage(imageData: data)
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
            } else {
                Text("No image selected")
                    .padding()
            }
            
            Spacer()
            
            Text("Voting Period: \(contestData.votingPeriod)")
            VStack {
                Text("Submit Fashion Image")
                    .font(.headline)
                    .padding()
                
                PhotosPicker("Select Image", selection: $selectedImageItem, matching: .images)
                    .buttonStyle(.borderedProminent)
            }
            .onChange(of: selectedImageItem) {
                Task {
                    if let data = try? await selectedImageItem?.loadTransferable(type: Data.self) {
                        if let uiImage = UIImage(data: data) {
                            displayedImage = Image(uiImage: uiImage)
                        }
                    }
                }
            }
        }
    }
    
    func postImage(imageData: Data) async {
        if let userId = UserDefaults.standard.string(forKey: "user_id") {
            print("User ID:", userId)
            if let submission = try? await networkManager.sendSubmission2(imageData: imageData, contestId: contestData.contestId, userId: userId) {
                if submission {
                    print("Submission successful. URL")
                }
                else {
                    print("error")
                }
            }
        }
        
    }
    
    
}

extension ContestViewData {
    static func previewValue(
        contestTitle: String = "Best Contest",
        contestDescription: String = "This is the best contest ever",
        submissionNumber: Int = 100,
        votingPeriod: DateInterval = .init(start: Date(), duration: 60)
    ) -> Self {
        .init(
            contestTitle: contestTitle,
            contestDescription: contestDescription,
            contestId: "4",
            votingPeriod: votingPeriod
        )
    }
}

#Preview {
    ContestView(contestData: .previewValue())
}
