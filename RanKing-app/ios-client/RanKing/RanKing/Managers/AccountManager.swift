import Foundation

class AccountManager: ObservableObject {
    @Published var isLoggedIn = false
    
    func login(email: String, password: String) {
        // TODO: call backend API
        self.isLoggedIn = true
    }
    
    func register(email: String, password: String) {
        // TODO: call backend API
    }
}
