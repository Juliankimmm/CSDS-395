import SwiftUI

struct ProfileHeader: View {
    let profile: UserProfile
    
    var body: some View {
        VStack(spacing: 12) {
            
            Image("avatar")
                .resizable()
                .frame(width: 90, height: 90)
                .clipShape(Circle())
            
            Text(profile.name)
                .font(.title2)
                .fontWeight(.semibold)
            
            Text(profile.handle)
                .foregroundColor(.gray)
                .font(.subheadline)
        }
    }
}
