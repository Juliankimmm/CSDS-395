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
    @State private var cameraImage: Image?
    @State private var capturedUIImage: UIImage?
    @State private var displayedImage: Image?
    @State private var showCamera: Bool = false
    
    @State private var didSelectImageFromCamera = false
    
    @Environment(\.dismiss) private var dismiss
    @State private var isSubmitting: Bool = false
    @State private var showSuccess: Bool = false

    var body: some View {
        ZStack {
            VStack {
                Text("Upload to \(contestData.contestTitle)")
                    .font(.system(size: 28, weight: .bold))
                    .padding(.top)
                Text(contestData.contestDescription)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                
                Group {
                    if let displayedImage {
                        displayedImage
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .shadow(radius: 5)
                            .padding()
                    }
                    Button("Submit") {
                        Task {
                            guard !isSubmitting else { return }
                            isSubmitting = true
                            var uploaded = false
                            if let data = try? await selectedImageItem?.loadTransferable(type: Data.self) {
                                await postImage(imageData: data)
                                uploaded = true
                            } else if didSelectImageFromCamera, let ui = capturedUIImage, let data = ui.jpegData(compressionQuality: 0.9) {
                                await postImage(imageData: data)
                                uploaded = true
                            }
                            await MainActor.run {
                                isSubmitting = false
                                if uploaded {
                                    withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                        showSuccess = true
                                    }
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                        dismiss()
                                    }
                                }
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSubmitting)
                }
                
                Spacer()
                
                VStack {
                    Text("Submit Fashion Image")
                        .font(.headline)
                        .padding()
                    
                    // Pick from photo library
                    PhotosPicker(
                        "Select Image",
                        selection: $selectedImageItem,
                        matching: .images
                    )
                    .buttonStyle(.borderedProminent)
                    
                    Button("Take Photo") {
                        showCamera = true
                    }
                    .buttonStyle(.borderedProminent)
                    .sheet(isPresented: $showCamera) {
                        CaptureImageView(isShown: $showCamera, image: $cameraImage, uiImage: $capturedUIImage)
                    }
                }
                .onChange(of: cameraImage) {
                    displayedImage = cameraImage
                    didSelectImageFromCamera = true
                }
                .onChange(of: selectedImageItem) {
                    Task {
                        if let data = try? await selectedImageItem?.loadTransferable(type: Data.self) {
                            if let uiImage = UIImage(data: data) {
                                displayedImage = Image(uiImage: uiImage)
                            }
                            didSelectImageFromCamera = false
                        }
                    }
                }
            }
            
            if showSuccess {
                VStack(spacing: 10) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 64, weight: .bold))
                        .foregroundStyle(.green)
                    Text("Uploaded!")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                .padding(24)
                .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(.ultraThinMaterial))
                .shadow(color: .black.opacity(0.2), radius: 16, y: 8)
                .transition(.opacity)
                .allowsHitTesting(false)
            }
            
            if isSubmitting {
                Color.black.opacity(0.08).ignoresSafeArea()
                ProgressView("Uploading...")
                    .padding()
                    .background(RoundedRectangle(cornerRadius: 14).fill(.ultraThinMaterial))
                    .shadow(radius: 8, y: 4)
            }
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.85), value: showSuccess)
    }
    
    func postImage(imageData: Data) async {
        if let userId = UserDefaults.standard.string(forKey: "user_id") {
            print("User ID:", userId)
            if let submission = try? await networkManager.sendSubmission2(imageData: imageData, contestId: contestData.contestId, userId: userId) {
                if submission {
                    print("Submission successful. URL")
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

