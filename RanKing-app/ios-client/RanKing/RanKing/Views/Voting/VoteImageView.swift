import SwiftUI

struct VotableImageView: View {
    
    let imageName: String
    let onVote: (SwipeDirection) -> Void

    @State private var offset: CGFloat = 0
    @State private var opacity: Double = 1

    var body: some View {
        Image(imageName)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .shadow(radius: 5)
            .offset(y: offset)
            .opacity(opacity)
            .gesture(
                DragGesture(minimumDistance: 20)
                    .onChanged { value in
                        offset = value.translation.height
                    }
                    .onEnded { value in
                        withAnimation(.spring()) {
                            if value.translation.height > 100 {
                                offset = 1000
                                opacity = 0
                                onVote(.down)
                            }
                            else if value.translation.height < -100 {
                                offset = -1000
                                opacity = 0
                                onVote(.up)
                            } else {
                                offset = 0
                            }
                        }
                    }
            )

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
    VotableImageView(imageName: "cutiepie") {
        voteDirection in
        print(voteDirection)
    }
}
