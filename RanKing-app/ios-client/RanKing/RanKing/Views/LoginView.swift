import SwiftUI

struct LoginView: View {
    
    let networkManager = NetworkManager.getInstance()
    
    @State private var email = ""
    @State private var password = ""
    @State private var showEmailLoginSheet: Bool = false
    
    @State private var isLoggedIn: Bool = false
    
    @State private var isConnectedToInternet: Bool = true

    var body: some View {
        if (isLoggedIn) {
            ContestListView()
        }
        else {
            ZStack {
                LinearGradient(
                    colors: [
                        Color.white,
                        Color(.systemGray6)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 28) {
                        RankedLogoWithDot()
                            .padding(.top, 60)
                            .phaseIn(0.0)
                            .bubbly()

                        

                        VStack(spacing: 8) {
                            Text("RanKing")
                                .font(.system(size: 36, weight: .semibold))
                                .foregroundColor(.black)

                            Text("Discover fashion through\ncommunity voting")
                                .multilineTextAlignment(.center)
                                .foregroundColor(Color(.darkGray))
                                .font(.system(size: 18))
                        }

                        HStack(spacing: 18) {
                            ForEach(Array(["fashion1", "fashion2", "fashion3"].enumerated()), id: \.element) { index, img in
                                Image(img)
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                                    .frame(width: 90, height: 130)
                                    .clipShape(RoundedRectangle(cornerRadius: 18))
                                    .shadow(color: Color.black.opacity(0.15), radius: 6, y: 3)
                                    .staggeredBubbly(Double(index) * 0.8)
                                    .phaseIn(Double(index) * 0.4) 
                            }
                        }
                        .padding(.top, 10)

                        Text("Join a community where style meets democracy. Vote on outfits, share your looks, and discover what's trending.")
                            .font(.system(size: 15))
                            .foregroundColor(Color(.gray))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)

                        Spacer().frame(height: 10)

                        Button(action: {
                            // TODO: Apple sign-in
                        }) {
                            Text("Continue with Apple")
                                .foregroundColor(.white)
                                .font(.system(size: 18, weight: .semibold))
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.black)
                                .cornerRadius(14)
                        }
                        .padding(.horizontal, 22)

                        Button(action: {
                            showEmailLoginSheet = true
                        }) {
                            Text("Sign up with Email")
                                .foregroundColor(.black)
                                .font(.system(size: 18, weight: .semibold))
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.white)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 14)
                                        .stroke(Color(.systemGray3), lineWidth: 1)
                                )
                        }
                        .padding(.horizontal, 22)

                        HStack {
                            Text("Already have an account?")
                                .foregroundColor(Color(.darkGray))

                            Button("Log in") {
                                showEmailLoginSheet = true
                            }
                            .foregroundColor(Color.blue)
                            .fontWeight(.semibold)
                        }
                        .padding(.bottom, 40)
                    }
                }

                VStack {
                    Spacer()
                    if !isConnectedToInternet {
                        Text("No internet connection")
                            .foregroundColor(Color.red)
                            .padding(.bottom, 12)
                    }
                }
            }
            .sheet(isPresented: $showEmailLoginSheet) {
                EmailPasswordLoginSheet(
                    email: $email,
                    password: $password,
                    onLogin: {
                        if email == "Admin" || password == "admin" {
                            isLoggedIn = true
                            return true
                        }
                        return false
                    },
                    doNetworkLogin: { email, password in
                        if let token = try? await networkManager.login(email: email, password: password) {
                            UserDefaults.standard.set(token.access_token, forKey: "token")
                            withAnimation(.smooth) {
                                isLoggedIn = true
                            }
                            return true
                        }
                        return false
                    }
                )
            }
            .onAppear() {
                Task {
                    isConnectedToInternet = try await networkManager.testServerIsRunning()
                }
            }
        }
    }
}

private struct EmailPasswordLoginSheet: View {
    @Binding var email: String
    @Binding var password: String
    var onLogin: () -> Bool
    var doNetworkLogin: (String, String) async -> Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                TextField("Email", text: $email)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                SecureField("Password", text: $password)
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                Button("Log in") {
                    if onLogin() {
                        dismiss()
                    } else {
                        Task {
                            let success = await doNetworkLogin(email, password)
                            if success { dismiss() }
                        }
                    }
                }
                .buttonStyle(.borderedProminent)

                Button("Cancel") {
                    dismiss()
                }
                .padding(.top, 4)

                Spacer()
            }
            .padding()
            .navigationTitle("Log in")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#Preview {
    LoginView()
}

struct PhaseInEffect: ViewModifier {
    @State private var appear = false
    
    var delay: Double
    
    func body(content: Content) -> some View {
        content
            .opacity(appear ? 1 : 0)
            .offset(y: appear ? 0 : 20)  // slight slide up
            .onAppear {
                withAnimation(
                    .easeOut(duration: 0.8)
                        .delay(delay)
                ) {
                    appear = true
                }
            }
    }
}

extension View {
    func phaseIn(_ delay: Double = 0) -> some View {
        self.modifier(PhaseInEffect(delay: delay))
    }
}
