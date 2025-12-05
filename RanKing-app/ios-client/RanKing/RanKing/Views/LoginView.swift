import SwiftUI

struct LoginView: View {
    
    let networkManager = NetworkManager.getInstance()
    
    @State private var email = "test@example.com"
    @State private var username = "alskdjhf"
    @State private var password = "secret123"
    @State private var showEmailLoginSheet: Bool = false
    @State private var showRegisterSheet: Bool = false
    
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
                                BlurFilledImage(named: img)
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
                            showRegisterSheet = true
                        }) {
                            Text("Sign up with Email")
                                .secondaryButton()
                        }
                        .padding(.horizontal, 22)

                        Button(action: {
                            showEmailLoginSheet = true
                        }) {
                            Text("Log in")
                                .primaryButton()
                        }
                        .padding(.horizontal, 22)

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
                    username: $username,
                    password: $password,
                    onLogin: {
                        if email == "Admin" || password == "admin" {
                            isLoggedIn = true
                            return true
                        }
                        return false
                    },
                    doNetworkLogin: { email, username, password in
                        if let token = try? await networkManager.login(email: email, password: password) {
                            // Username currently unused by backend login; stored locally if needed
                            UserDefaults.standard.set(token, forKey: "user_id")
                            withAnimation(.smooth) {
                                isLoggedIn = true
                            }
                            return true
                        }
                        return false
                    }
                )
            }
            .sheet(isPresented: $showRegisterSheet) {
                RegisterView()
            }
            .onAppear() {
                Task {
                    isConnectedToInternet = try await networkManager.testServerIsRunning()
                }
            }
        }
    }
}

private struct BlurFilledImage: View {
    let image: Image

    init(named name: String) {
        self.image = Image(name)
    }

    init(_ image: Image) {
        self.image = image
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Blurred background to fill any letterboxing areas
                image
                    .resizable()
                    .scaledToFill()
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
                    .blur(radius: 20)
                    .saturation(1.2)
                    .opacity(0.9)

                // Foreground image with aspect fill
                image
                    .resizable()
                    .scaledToFill()
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
            }
        }
    }
}

private struct EmailPasswordLoginSheet: View {
    @Binding var email: String
    @Binding var username: String
    @Binding var password: String
    var onLogin: () -> Bool
    var doNetworkLogin: (String, String, String) async -> Bool
    @Environment(\.dismiss) private var dismiss
    @State private var isSubmitting: Bool = false
    @State private var errorMessage: String? = nil

    var body: some View {
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
                VStack(spacing: 24) {
                    VStack(spacing: 8) {
                        Text("Welcome back")
                            .font(.system(size: 32, weight: .semibold))
                            .foregroundColor(.primary)
                            .phaseIn(0)
                        Text("Log in to continue")
                            .font(.system(size: 16))
                            .foregroundColor(Color(.darkGray))
                            .phaseIn(0.15)
                    }
                    .padding(.top, 24)

                    VStack(spacing: 14) {
                        TextField("Email", text: $email)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .phaseIn(0.2)

                        SecureField("Password", text: $password)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .phaseIn(0.4)
                    }
                    .padding(.horizontal, 22)

                    if let error = errorMessage, !error.isEmpty {
                        Text(error)
                            .font(.footnote)
                            .foregroundColor(.red)
                            .padding(.horizontal, 22)
                    }

                    VStack(spacing: 12) {
                        Button(action: {
                            if onLogin() {
                                dismiss()
                            } else {
                                isSubmitting = true
                                errorMessage = nil
                                Task {
                                    let success = await doNetworkLogin(email, username, password)
                                    await MainActor.run {
                                        isSubmitting = false
                                        if success {
                                            dismiss()
                                        } else {
                                            errorMessage = "Invalid credentials. Please try again."
                                        }
                                    }
                                }
                            }
                        }) {
                            HStack {
                                if isSubmitting { ProgressView().tint(.white) }
                                Text("Log in")
                            }
                            .primaryButton()
                        }
                        .padding(.horizontal, 22)
                        .disabled(email.isEmpty || username.isEmpty || password.isEmpty || isSubmitting)

                        Button(action: { dismiss() }) {
                            Text("Cancel")
                                .secondaryButton()
                        }
                        .padding(.horizontal, 22)
                    }
                    .padding(.bottom, 24)
                }
            }
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

private struct PrimaryButtonStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.system(size: 18, weight: .semibold))
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                LinearGradient(colors: [Color.blue, Color.purple], startPoint: .topLeading, endPoint: .bottomTrailing)
            )
            .cornerRadius(16)
            .shadow(color: Color.black.opacity(0.12), radius: 10, y: 5)
            .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .opacity(1)
    }
}

private struct SecondaryButtonStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.system(size: 18, weight: .semibold))
            .foregroundColor(.primary)
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemBackground))
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(Color(.systemGray3), lineWidth: 1)
            )
            .cornerRadius(16)
            .shadow(color: Color.black.opacity(0.06), radius: 6, y: 3)
            .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

private extension View {
    func primaryButton() -> some View { self.modifier(PrimaryButtonStyle()) }
    func secondaryButton() -> some View { self.modifier(SecondaryButtonStyle()) }
}

extension View {
    func phaseIn(_ delay: Double = 0) -> some View {
        self.modifier(PhaseInEffect(delay: delay))
    }
}

