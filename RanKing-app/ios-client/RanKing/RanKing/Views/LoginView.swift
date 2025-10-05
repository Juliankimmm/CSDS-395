import SwiftUI

struct LoginView: View {
    @State private var email = ""
    @State private var password = ""
    
    
    @State private var isLoggedIn: Bool = false

    var body: some View {
        if (isLoggedIn) {
            ContestListView()
        }
        else {
            VStack {
                Text("RanKing Login")
                    .font(.largeTitle)
                TextField("Email", text: $email)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                SecureField("Password", text: $password)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                Button("Login") {
                    // TODO: call backend API
                    withAnimation(.smooth) {
                        isLoggedIn = true
                    }
                }
                .padding()
            }
            .padding()
        }
    }
}

#Preview {
    LoginView()
}
