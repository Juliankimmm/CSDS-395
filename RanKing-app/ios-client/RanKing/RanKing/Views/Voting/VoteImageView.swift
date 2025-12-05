import SwiftUI

struct VotableImageView: View {
    
    let image : Image
    
    @Binding var scale : CGFloat
    @State private var xOffset: CGFloat = 0
    @State private var yOffset: CGFloat = 0
    @State private var opacity: Double = 1
    
    let onVote: (SwipeDirection) -> Void

    var body: some View {
        ZStack {
            image
                .resizable()
                .aspectRatio(contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .shadow(radius: 5)
                .offset(x: xOffset, y: yOffset)
                .opacity(opacity)
                .scaleEffect(scale)
                .gesture(
                    DragGesture(minimumDistance: 20)
                        .onChanged { value in
                            yOffset = value.translation.height
                            xOffset = value.translation.width
                        }
                        .onEnded { value in
                            withAnimation(.spring()) {
                                if value.translation.height > 200 {
                                    yOffset = 1000
                                    opacity = 0
                                    onVote(.down)
                                }
                                else if value.translation.height < -200 {
                                    yOffset = -1000
                                    opacity = 0
                                    onVote(.up)
                                }
                                else if value.translation.width > 100 {
                                    xOffset = 1000
                                    opacity = 0
                                    onVote(.up)
                                }
                                else if value.translation.width < -100 {
                                    xOffset = -1000
                                    opacity = 0
                                    onVote(.up)
                                }
                                else {
                                    yOffset = 0
                                    xOffset = 0
                                }
                            }
                        }
                )
        }
        .onAppear {
            Task {
                
            }
        }

    }
}

enum SwipeDirection {
    case up
    case down
}

enum ImagePosition {
    case top
    case bottom
}

#Preview {
    struct VotableImageView_Preview_Host: View {
        
        // 2. Put the @State variable here, where it's a valid property of a View.
        @State var previewImage: Image = Image("jezthisguyishot")
        @State var scale: CGFloat = 1.0

        var body: some View {
            // 3. Initialize your component inside the host's body.
            VotableImageView(image: previewImage, scale: $scale) { voteDirection in
                print("Voted:", voteDirection)
            }
        }
    }
    
    // 4. Return an instance of the host view.
    return VotableImageView_Preview_Host()
}
