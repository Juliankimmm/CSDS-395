import SwiftUI

struct RegisterView: View {
    
    let networkManager = NetworkManager.getInstance()
    
    @State private var email = ""
    @State private var username = ""
    @State private var password = ""
    

    var body: some View {
        VStack {
            Text("RanKing Registration")
                .font(.largeTitle)
            TextField("Email", text: $email)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            TextField("Username", text: $username)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            SecureField("Password", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            Button("Register") {
                Task {
                    try? await networkManager.register(username: username, email: email, password: password)
                }
            }
            .buttonStyle(.borderedProminent)
            .padding()
        }
        .padding()
    }
}

#Preview {
    RegisterView()
}
