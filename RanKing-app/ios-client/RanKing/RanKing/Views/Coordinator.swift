//
//  Coordinator.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 12/5/25.
//


import SwiftUI

struct ContentView: View {
    @State var image: Image? = nil
    @State var showCaptureImageView: Bool = false

    var body: some View {
        ZStack {
            VStack {
                Button(action: {
                   self.showCaptureImageView.toggle()
                }) {
                    Text("Choose photos")
                }
                image?.resizable()
                  .frame(width: 250, height: 250)
                  .clipShape(Circle())
                  .overlay(Circle().stroke(Color.white, lineWidth: 4))
                  .shadow(radius: 10)
            }
            if (showCaptureImageView) {
//                    CaptureImageView(isShown: $showCaptureImageView, image: $image)
            }
        }
    }
}

struct CaptureImageView {
    @Binding var isShown: Bool
    @Binding var image: Image?
    @Binding var uiImage: UIImage?

    func makeCoordinator() -> Coordinator {
      return Coordinator(isShown: $isShown, image: $image, uiImage: $uiImage)
    }
}

extension CaptureImageView: UIViewControllerRepresentable {
    func makeUIViewController(context: UIViewControllerRepresentableContext<CaptureImageView>) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController,
                                context: UIViewControllerRepresentableContext<CaptureImageView>) {

    }
}

class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
  @Binding var isCoordinatorShown: Bool
  @Binding var imageInCoordinator: Image?
  @Binding var uiImage: UIImage?
    
  init(isShown: Binding<Bool>, image: Binding<Image?>, uiImage: Binding<UIImage?>) {
    _isCoordinatorShown = isShown
    _imageInCoordinator = image
    _uiImage = uiImage
  }

  func imagePickerController(_ picker: UIImagePickerController,
                didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
     guard let unwrapImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else { return }
     imageInCoordinator = Image(uiImage: unwrapImage)
     uiImage = unwrapImage
     isCoordinatorShown = false
  }

  func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
     isCoordinatorShown = false
  }
}

#Preview {
    ContentView()
}
