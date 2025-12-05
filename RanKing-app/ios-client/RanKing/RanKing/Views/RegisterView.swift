import SwiftUI

struct RegisterView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var email: String = ""
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
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
                    // Title & subtitle
                    VStack(spacing: 8) {
                        Text("Create account")
                            .font(.system(size: 32, weight: .semibold))
                            .foregroundColor(.primary)
                            .phaseIn(0)
                        Text("Join the RanKing community")
                            .font(.system(size: 16))
                            .foregroundColor(Color(.darkGray))
                            .phaseIn(0.15)
                    }
                    .padding(.top, 40)

                    // Fields
                    VStack(spacing: 14) {
                        TextField("Email", text: $email)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .phaseIn(0.2)

                        TextField("Username", text: $username)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .phaseIn(0.3)

                        SecureField("Password", text: $password)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .phaseIn(0.4)

                        SecureField("Confirm password", text: $confirmPassword)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .phaseIn(0.5)
                    }
                    .padding(.horizontal, 22)

                    if let error = errorMessage, !error.isEmpty {
                        Text(error)
                            .font(.footnote)
                            .foregroundColor(.red)
                            .padding(.horizontal, 22)
                    }

                    // Actions
                    VStack(spacing: 12) {
                        Button(action: submit) {
                            HStack {
                                if isSubmitting { ProgressView().tint(.white) }
                                Text("Sign up")
                            }
                            .primaryButton()
                        }
                        .padding(.horizontal, 22)
                        .disabled(!isFormValid || isSubmitting)

                        Button(action: { dismiss() }) {
                            Text("Back to login")
                                .secondaryButton()
                        }
                        .padding(.horizontal, 22)
                    }
                    .padding(.bottom, 30)
                }
            }
        }
        .navigationBarBackButtonHidden(true)
    }

    private var isFormValid: Bool {
        guard !email.isEmpty, !username.isEmpty, !password.isEmpty, !confirmPassword.isEmpty else { return false }
        guard password == confirmPassword else { return false }
        return true
    }

    private func submit() {
        errorMessage = nil
        guard isFormValid else {
            errorMessage = "Please fill all fields and ensure passwords match."
            return
        }
        isSubmitting = true
        // TODO: Wire to NetworkManager registration when available
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isSubmitting = false
            dismiss()
        }
    }
}

#Preview {
    RegisterView()
}

// Lightweight copies of the button styles to keep this file self-contained.
private struct PrimaryButtonStyle_Register: ViewModifier {
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
    }
}

private struct SecondaryButtonStyle_Register: ViewModifier {
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
    func primaryButton() -> some View { self.modifier(PrimaryButtonStyle_Register()) }
    func secondaryButton() -> some View { self.modifier(SecondaryButtonStyle_Register()) }
}
