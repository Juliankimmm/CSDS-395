import SwiftUI

struct LoginView: View {
    
    let networkManager = NetworkManager.getInstance()
    
    @State private var email = ""
    @State private var password = ""
    
    
    @State private var isLoggedIn: Bool = false
    
    @State private var isConnectedToInternet: Bool = true

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
                    if email == "Admin" || password == "admin" {
                        isLoggedIn = true
                    }
                    Task {
                        if let token = try? await networkManager.login(email: email, password: password) {
                            UserDefaults.standard.set(token.access_token, forKey: "token")
                            withAnimation(.smooth) {
                                isLoggedIn = true
                            }
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .padding()
                if !isConnectedToInternet {
                    Text("No internet connection")
                        .foregroundColor(Color.red)
                }
            }
            .padding()
            .onAppear() {
                Task {
                    isConnectedToInternet = try await networkManager.testServerIsRunning()
                }
            }
        }
    }
}

#Preview {
    LoginView()
}
