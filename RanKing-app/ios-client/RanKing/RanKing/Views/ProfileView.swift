
import SwiftUI

struct ProfileView: View {
    @State private var email = ""
    @State private var password = ""

    var body: some View {
        VStack {
            Text("Profile View")
                .font(.largeTitle)
                .padding()
        }
        .padding()
    }
}

#Preview {
    ProfileView()
}
